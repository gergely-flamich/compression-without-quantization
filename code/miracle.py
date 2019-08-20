import argparse
import glob
import os

from tqdm import tqdm

from absl import app
from absl.flags import argparse_flags

from pln import ProbabilisticLadderNetwork, VariationalAutoEncoder

import tensorflow.compat.v1 as tf

import numpy as np

def read_png(filename):
    """
    Loads a PNG image file. Taken from Balle's implementation
    """
    image_raw = tf.io.read_file(filename)
    image = tf.image.decode_image(image_raw, channels=3)
    image = tf.cast(image, tf.float32)
    image /= 255.
    
    return image

def quantize_image(image):
    """
    Taken from Balle's implementation
    """
    image = tf.round(image * 255)
    image = tf.saturate_cast(image, tf.uint8)
    return image


def write_png(filename, image):
    """
    Saves an image to a PNG file. Taken from Balle's implementation
    """
    image = quantize_image(image)
    string = tf.image.encode_png(image)
    return tf.write_file(filename, string)

def create_dataset(dataset_path, patch_size, batch_size, preprocess_threads, file_regex="*.png"):
    """
    Based on Balle's dataset creation method
    """
    # Create input data pipeline.
    with tf.device("/CPU:0"):
        
        train_files = glob.glob(dataset_path + "/" + file_regex)
        
        if not train_files:
            raise RuntimeError("No training images found at '{}'.".format(dataset_path ))
            
        train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
        train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).repeat()
        train_dataset = train_dataset.map(
            read_png, num_parallel_calls=preprocess_threads)
        train_dataset = train_dataset.map(
            lambda x: tf.image.random_crop(x, (patch_size, patch_size, 3)))
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(32)
        
    return train_dataset


def train(args):
    
    # ------------------------------------------------------------------------
    # Load Dataset
    # ------------------------------------------------------------------------
    train_dataset = create_dataset(args.data_path, 
                                   args.patch_size, 
                                   args.batch_size, 
                                   args.preprocess_threads)
    
    num_pixels = args.batch_size * args.patch_size ** 2
    
    batch = train_dataset.make_one_shot_iterator().get_next()
    
    # ------------------------------------------------------------------------
    # Create Model and optimizer
    # ------------------------------------------------------------------------
    
    if args.model == "pln":
        model = ProbabilisticLadderNetwork(first_level_filters=196,
                                           second_level_filters=196,
                                           first_level_latent_channels=128,
                                           second_level_latent_channels=24,
                                           likelihood=args.likelihood,
                                           learn_gamma=args.learn_gamma)

    elif args.model == "vae":
        model = VariationalAutoEncoder(num_filters=128,
                                       num_latent_channels=128,
                                       likelihood=args.likelihood,
                                       learn_gamma=args.learn_gamma)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
            
    # ------------------------------------------------------------------------
    # Set up Loss
    # ------------------------------------------------------------------------ 
    step = tf.train.create_global_step()
    
    reconstructions = model(batch)

    warmup_coef = tf.minimum(1., tf.cast(step / args.warmup_steps, tf.float32))

    # The multiplication is at the end is to compensate for the rescaling
    neg_log_likelihood = -tf.reduce_mean(model.log_likelihood) * 255**2

    bpp_regularizer = warmup_coef * args.beta * model.bpp(num_pixels)

    train_loss = neg_log_likelihood + bpp_regularizer
    
    train_step = optimizer.minimize(train_loss, global_step=step)

    # ------------------------------------------------------------------------
    # Tensorboard Summaries
    # ------------------------------------------------------------------------ 
    tf.summary.scalar("Loss", train_loss)
    tf.summary.scalar("Log-Probability", -neg_log_likelihood)
    tf.summary.scalar("Rate-Regularization", bpp_regularizer)
    tf.summary.scalar("Bits-Per-Pixel", model.bpp(num_pixels))
    tf.summary.scalar("Warmup-Coeff", warmup_coef)
    tf.summary.scalar("Average-PSNR", tf.reduce_sum(tf.image.psnr(batch, 
                                                   tf.clip_by_value(reconstructions, 0., 1.),
                                                   max_val=1.0)) / args.batch_size)
    tf.summary.image("Reconstruction", quantize_image(reconstructions))
    tf.summary.image("Original", quantize_image(batch))

    if args.learn_gamma:
        tf.summary.scalar("Gamma", tf.exp(model.log_gamma))

    if args.model == "pln":
        for i, level_kl_divs in enumerate([model.first_level_kl, model.second_level_kl]): 
            tf.summary.scalar("Max-KL-on-Level-{}".format(i + 1), tf.reduce_max(level_kl_divs))
            
    elif args.model == "vae":
        tf.summary.scalar("Max-KL", tf.reduce_max(model.kl_divergence))
        
    hooks = [
        tf.train.StopAtStepHook(last_step=args.train_steps),
        tf.train.NanTensorHook(train_loss),
    ]
    with tf.train.MonitoredTrainingSession(
        hooks=hooks, checkpoint_dir=args.model_dir,
        save_checkpoint_secs=args.checkpoint_freq, save_summaries_secs=args.log_freq) as sess:
        while not sess.should_stop():
            sess.run(train_step)

        
# ============================================================================
# ============================================================================
# Plumbing
# ============================================================================
# ============================================================================


def parse_args(argv):
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--verbose", "-V", action="store_true",
                        help="Turn logging on")
    
    parser.add_argument("--model_dir", "-M", 
                        help="Model directory where we will save the checkpoints and Tensorboard logs")
    
    subparsers = parser.add_subparsers(title="mode",
                                       dest="mode",
                                       help="Current available modes: train")
    
    # Training mode
    train_mode = subparsers.add_parser("train",
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                       description="Train a new model")
    
    train_mode.add_argument("--data_path", "-D",
                            help="Path to PNG dataset")
    train_mode.add_argument("--model", default="vae",
                            help="Model to train: vae or pln")
    train_mode.add_argument("--patch_size", default=256, type=int,
                            help="Square patch size for the random crops")
    train_mode.add_argument("--batch_size", default=8, type=int,
                            help="Number of image patches per training batch for SGD")
    train_mode.add_argument("--preprocess_threads", default=16, type=int,
                            help="Number of threads to use to process the dataset")
    train_mode.add_argument("--checkpoint_freq", default=600, type=int,
                            help="Checkpointing frequency (seconds)")
    train_mode.add_argument("--log_freq", default=30, type=int,
                            help="Logging frequency (seconds)")
    train_mode.add_argument("--train_steps", default=200000, type=int,
                            help="Number of training iterations")
    train_mode.add_argument("--likelihood", default="gaussian",
                            help="Gaussian or Laplace")
    train_mode.add_argument("--beta", default=10, type=float,
                            help="KL coefficient in the training loss")
    train_mode.add_argument("--learning_rate", default=0.001, type=float,
                            help="Learning rate")
    train_mode.add_argument("--warmup_steps", default=40000, type=int,
                            help="Number of warmup steps for the KL coefficient")
    train_mode.add_argument("--learn_gamma", action="store_true", default=False,
                            help="Turns on the gamma learning technique suggested by Dai and Wipf")
    
    # Parse arguments
    args = parser.parse_args(argv[1:])
    if args.mode is None:
        parser.print_usage()
        sys.exit(2)
        
    return args


def main(args):
    if args.mode == "train":
        train(args)

if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
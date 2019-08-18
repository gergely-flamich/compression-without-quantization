import argparse
import glob
import os

from tqdm import tqdm

from absl import app
from absl.flags import argparse_flags

from pln import ProbabilisticLadderNetwork

import tensorflow as tf
tfs = tf.contrib.summary
tfs_logger = tfs.record_summaries_every_n_global_steps
tf.compat.v1.enable_eager_execution()

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
    
    # ------------------------------------------------------------------------
    # Create Model and optimizer
    # ------------------------------------------------------------------------
    pln = ProbabilisticLadderNetwork(first_level_filters=196,
                                     second_level_filters=128,
                                     first_level_latent_channels=128,
                                     second_level_latent_channels=24,
                                     learn_gamma=args.learn_gamma)
    
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)

    # ------------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------------
    global_step = tf.compat.v1.train.get_or_create_global_step()
    
    checkpoint_prefix = args.model_dir + "/ckpt"
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     model=pln,
                                     optimizer_step=global_step)
    
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=5)
    
    checkpoint.restore(manager.latest_checkpoint)
    
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    
    # ------------------------------------------------------------------------
    # Tensorboard stuff
    # ------------------------------------------------------------------------
    logdir = args.model_dir + "/log"
    writer = tfs.create_file_writer(logdir)
    writer.set_as_default()
    
    # ------------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------------
    for batch in tqdm(train_dataset.take(args.train_steps), total=args.train_steps):
        
        # Increment global step
        global_step.assign_add(1)
        
        # Create loss
        with tf.GradientTape() as tape, tfs_logger(args.log_freq):
            
            # Pass batch through the PLN
            reconstructions = pln(batch)
            
            warmup_coef = tf.minimum(1., global_step.numpy() / args.warmup_steps)

            # The multiplication is at the end is to compensate for the rescaling
            neg_log_likelihood = -tf.reduce_mean(pln.log_likelihood) * 255**2
            
            bits_per_pixel = (tf.reduce_sum(pln.first_level_kl) +
                              tf.reduce_sum(pln.second_level_kl)) / (np.log(2) * num_pixels)

            bpp_regularizer = warmup_coef * args.beta * bits_per_pixel
            
            loss = neg_log_likelihood + bpp_regularizer

            # Add tensorboard summaries
            tfs.scalar("Loss", loss)
            tfs.scalar("Log-Probability", neg_log_likelihood)
            tfs.scalar("Rate-Regularization", bpp_regularizer)
            tfs.scalar("Bits-Per-Pixel", bits_per_pixel)
            tfs.scalar("Warmup-Coeff", warmup_coef)
            tfs.scalar("Average-PSNR", tf.reduce_sum(tf.image.psnr(batch, 
                                                                   tf.clip_by_value(reconstructions, 0., 1.),
                                                                   max_val=1.0)) / args.batch_size)
            tfs.image("Reconstruction", quantize_image(reconstructions))
            tfs.image("Original", quantize_image(batch))
            
            if args.learn_gamma:
                tfs.scalar("Gamma", tf.exp(pln.log_gamma))

            for i, level_kl_divs in enumerate([pln.first_level_kl, pln.second_level_kl]): 
                tfs.scalar("Max-KL-on-Level-{}".format(i + 1), tf.reduce_max(level_kl_divs))

        # Backprop
        grads = tape.gradient(loss, pln.trainable_variables)
        optimizer.apply_gradients(zip(grads, pln.trainable_variables))

        if int(global_step) % args.checkpoint_freq == 0:
            print("Saving model")
            manager.save()

        
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
    train_mode.add_argument("--patch_size", default=256, type=int,
                            help="Square patch size for the random crops")
    train_mode.add_argument("--batch_size", default=8, type=int,
                            help="Number of image patches per training batch for SGD")
    train_mode.add_argument("--preprocess_threads", default=16, type=int,
                            help="Number of threads to use to process the dataset")
    train_mode.add_argument("--checkpoint_freq", default=5000, type=int,
                            help="Checkpointing frequency")
    train_mode.add_argument("--log_freq", default=50, type=int,
                            help="Logging frequency")
    train_mode.add_argument("--train_steps", default=200000, type=int,
                            help="Number of training iterations")
    train_mode.add_argument("--beta", default=0.1, type=float,
                            help="KL coefficient in the training loss")
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
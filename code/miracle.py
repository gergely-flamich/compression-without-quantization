import argparse
import glob
import os
import json
import time

from tqdm import tqdm

from absl import app
from absl.flags import argparse_flags

from pln import ProbabilisticLadderNetwork
from vae import VariationalAutoEncoder

import tensorflow.compat.v1 as tf

import numpy as np

import matplotlib.pyplot as plt

# ============================================================================
# ============================================================================
# Helper functions
# ============================================================================
# ============================================================================

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

# ============================================================================
# ============================================================================
# Model Training
# ============================================================================
# ============================================================================

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
        model = ProbabilisticLadderNetwork(first_level_filters=args.filters1,
                                           second_level_filters=args.filters2,
                                           first_level_latent_channels=args.latent_channels1,
                                           second_level_latent_channels=args.latent_channels2,
                                           likelihood=args.likelihood,
                                           learn_gamma=args.learn_gamma)

    elif args.model == "vae":
        model = VariationalAutoEncoder(num_filters=args.filters,
                                       num_latent_channels=args.latent_channels,
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
# Compresssion
# ============================================================================
# ============================================================================

def compress(args):
    
    # Load input image and add batch dimension.
    image = read_png(args.input_file)
#     image = tf.image.random_crop(image, [128, 128, 3])
    image = tf.expand_dims(image, 0)
    image.set_shape([1, None, None, 3])
    image_shape = tf.shape(image)
    
    if args.model == "pln":
        model = ProbabilisticLadderNetwork(first_level_filters=args.filters1,
                                           second_level_filters=args.filters2,
                                           first_level_latent_channels=args.latent_channels1,
                                           second_level_latent_channels=args.latent_channels2,
                                           likelihood="gaussian", # These doesn't matter for compression
                                           learn_gamma=True)
        
    elif args.model == "vae":
        model = VariationalAutoEncoder(num_filters=args.filters,
                                       num_latent_channels=args.latent_channels,
                                       likelihood="gaussian",
                                       learn_gamma=True)
        
        
    reconstruction = model(image)
        
    with tf.Session() as sess:
        # Load the latest model checkpoint, get the compressed string and the tensor
        # shapes.
        latest = tf.train.latest_checkpoint(checkpoint_dir=args.model_dir)
        saver = tf.train.import_meta_graph(latest + '.meta', clear_devices=True)
        saver.restore(sess, latest)
        
#         sess.run(reconstruction)
        
#         q1_loc = model.posterior_1.loc
#         q1_scale = model.posterior_1.scale
#         p1_loc = model.prior_1.loc
#         p1_scale = model.prior_1.scale
        
#         q1_loc, q1_scale, p1_loc, p1_scale = sess.run([q1_loc, q1_scale, p1_loc, p1_scale])
        
#         model_dir = "/scratch/gf332/data/kodak_cwoq/"
        
#         np.save(model_dir + "pln_q1_loc.npy", q1_loc)
#         np.save(model_dir + "pln_q1_scale.npy", q1_scale)
#         np.save(model_dir + "pln_p1_loc.npy", p1_loc)
#         np.save(model_dir + "pln_p1_scale.npy", p1_scale)
        
#         return
        
        model.code_image_greedy(session=sess,
                                image=image, 
                                seed=args.seed, 
                                n_steps=args.n_steps,
                                n_bits_per_step=args.n_bits_per_step,
                                comp_file_path=args.output_file,
                                first_level_max_group_size_bits=args.first_level_max_group_size_bits,
                                use_importance_sampling=True,
                                second_level_n_bits_per_group=args.second_level_n_bits_per_group,
                                second_level_max_group_size_bits=args.second_level_max_group_size_bits,
                                second_level_dim_kl_bit_limit=args.second_level_dim_kl_bit_limit,
                                outlier_index_bytes=args.outlier_index_bytes,
                                outlier_sample_bytes=args.outlier_sample_bytes,
                                verbose=args.verbose)

        
# ============================================================================
# ============================================================================
# Decompresssion
# ============================================================================
# ============================================================================

def decompress(args):
    
    if args.model == "pln":
        model = ProbabilisticLadderNetwork(first_level_filters=args.filters1,
                                           second_level_filters=args.filters2,
                                           first_level_latent_channels=args.latent_channels1,
                                           second_level_latent_channels=args.latent_channels2,
                                           likelihood="gaussian", # These doesn't matter for compression
                                           learn_gamma=True)
        
    elif args.model == "vae":
        model = VariationalAutoEncoder(num_filters=args.filters,
                                       num_latent_channels=args.latent_channels,
                                       likelihood="gaussian",
                                       learn_gamma=True)
        
    r = model(tf.zeros((1, 128, 128, 3)))
    with tf.Session() as sess:
        # Load the latest model checkpoint, get the compressed string and the tensor
        # shapes.
        latest = tf.train.latest_checkpoint(checkpoint_dir=args.model_dir)
        saver = tf.train.import_meta_graph(latest + '.meta', clear_devices=True)
        saver.restore(sess, latest)
        sess.run(r)
        
        reconstruction = model.decode_image_greedy(session=sess,
                                                  comp_file_path=args.comp_file,
                                                  verbose=args.verbose)
        
        sess.run(write_png(args.output_file, reconstruction))

        
# ============================================================================
# ============================================================================
# Get compression stats for a whole dataset
# ============================================================================
# ============================================================================

def compress_dataset(args,
                     dataset_im_format="/kodim{:02d}.png",
                     comp_file_format="/kodim{:02d}.miracle",
                     n_images=24):
    
#     dataset = tf.data.Dataset.range(100)
#     iterator = dataset.make_one_shot_iterator()
#     next_element = iterator.get_next()

#     with tf.Session() as sess:
#         for i in range(100):
#             value = sess.run(next_element)
#             assert i == value
    
#     return
    
    reconstruction_root = args.reconstruction_root
    reconstruction_subdir = args.reconstruction_subdir
    
    # ========================================================================
    # Create datasets
    # ========================================================================
    
    if args.theoretical:
        reconstruction_subdir = "theoretical_" + reconstruction_subdir

    reconstruction_path = reconstruction_root + "/" + reconstruction_subdir

    if not os.path.exists(reconstruction_path):
        print("Creating reconstruction directory " + reconstruction_path)

        os.makedirs(reconstruction_path)

    # Create lists of paths for every image in the dataset
    dataset_im_paths = [args.dataset_path + "/" + dataset_im_format.format(i) 
                      for i in range(1, n_images + 1)]

    reconstruction_im_paths = [reconstruction_path + "/" + dataset_im_format.format(i) 
                               for i in range(1, n_images + 1)]

    comp_file_paths = [reconstruction_path + "/" + comp_file_format.format(i) 
                       for i in range(1, n_images + 1)]

    # Load in the dataset
    if not dataset_im_paths:
        raise RuntimeError("No training images found at '{}'.".format(args.dataset_path ))

    paths_ds = tf.data.Dataset.from_tensor_slices(dataset_im_paths)
    image_ds = paths_ds.map(
        read_png, num_parallel_calls=16)
    image_ds = image_ds.prefetch(32)
    
    image = image_ds.make_one_shot_iterator().get_next()
    # ========================================================================
    # Reload model
    # ========================================================================
    
    if args.model == "pln":
        model = ProbabilisticLadderNetwork(first_level_filters=args.filters1,
                                           second_level_filters=args.filters2,
                                           first_level_latent_channels=args.latent_channels1,
                                           second_level_latent_channels=args.latent_channels2,
                                           likelihood="gaussian", # These doesn't matter for compression
                                           learn_gamma=True)
        
    elif args.model == "vae":
        model = VariationalAutoEncoder(num_filters=args.filters,
                                       num_latent_channels=args.latent_channels,
                                       likelihood="gaussian",
                                       learn_gamma=True)
        
        
    reconstruction = model(tf.zeros((1, 256, 256, 3)))
        
    with tf.Session() as sess:
        # Load the latest model checkpoint, get the compressed string and the tensor
        # shapes.
        
        latest = tf.train.latest_checkpoint(checkpoint_dir=args.model_dir)
        saver = tf.train.import_meta_graph(latest + '.meta', clear_devices=True)
        saver.restore(sess, latest)
        
        next_image = image_ds.make_one_shot_iterator().get_next()

        next_image = tf.expand_dims(image, 0)
        next_image.set_shape([1, None, None, 3])

        for i in range(n_images):
            dataset_im_name = dataset_im_format.format(i + 1)
            stats_path = reconstruction_root + "/stats.json"
            
            image = sess.run(next_image)

            # Everything is sampled from the true posterior
            if args.theoretical:
                reconstruction = sess.run(model(image))
                
                summaries = {}
                
                encoding_time = -1
                decoding_time = -1
                
                total_kl = sess.run(tf.reduce_sum(model.first_level_kl) + tf.reduce_sum(model.second_level_kl))
                theoretical_byte_size = (total_kl + 2 * np.log(total_kl + 1)) / np.log(2)

                image_shape = image.shape

                bpp = theoretical_byte_size / (image_shape[1] * image_shape[2]) 
                
                summaries = {"bpp": float(bpp),
                             "encoding_time": encoding_time,
                             "decoding_time": decoding_time}

            # Non-theoretical reconstruction
            else:
                if os.path.exists(comp_file_paths[i]):
                    print(comp_file_paths[i] + " already exists, skipping coding.")

                else:

                    start_time = time.time()
                    _, summaries = model.code_image_greedy(session=sess,
                                                        image=image, 
                                                        seed=args.seed, 
                                                        n_steps=args.n_steps,
                                                        n_bits_per_step=args.n_bits_per_step,
                                                        comp_file_path=comp_file_paths[i],
                                                        first_level_max_group_size_bits=args.first_level_max_group_size_bits,
                                                        use_importance_sampling=True,
                                                        second_level_n_bits_per_group=args.second_level_n_bits_per_group,
                                                        second_level_max_group_size_bits=args.second_level_max_group_size_bits,
                                                        second_level_dim_kl_bit_limit=args.second_level_dim_kl_bit_limit,
                                                        outlier_index_bytes=args.outlier_index_bytes,
                                                        outlier_sample_bytes=args.outlier_sample_bytes,
                                                        verbose=args.verbose)

                    encoding_time = time.time() - start_time

                if os.path.exists(reconstruction_im_paths[i]):
                    print(reconstruction_im_paths[i] + " already exists, skipping reconstruction.")

                else:
                    start_time = time.time()
                    reconstruction = model.decode_image_greedy(session=sess,
                                                             comp_file_path=comp_file_paths[i],
                                                             verbose=args.verbose,
                                                             rho=1.)
                    decoding_time = time.time() - start_time
                    print("Writing " + reconstruction_im_paths[i])

            if args.theoretical or not os.path.exists(reconstruction_im_paths[i]):
                ms_ssim = tf.image.ssim_multiscale(image, reconstruction, max_val=1.0)
                psnr = tf.image.psnr(image, reconstruction, max_val=1.0)    

                ms_ssim, psnr = sess.run([ms_ssim, psnr])

                if not os.path.exists(reconstruction_im_paths[i]):
                    sess.run(write_png(reconstruction_im_paths[i], tf.squeeze(reconstruction)))

                summaries["image_shape"] = summaries["image_shape"].tolist()
                summaries["encoding_time"] = encoding_time
                summaries["decoding_time"] = decoding_time
                summaries["total_time"] = encoding_time + decoding_time
                summaries["ms_ssim"] = float(ms_ssim)
                summaries["psnr"] = float(psnr)

                print(summaries)

                if os.path.exists(stats_path):
                    with open(stats_path, "r") as stats_fp:
                        stats = json.load(stats_fp)
                else:
                    stats = {}

                if dataset_im_name not in stats:
                    stats[dataset_im_name] = {}

                with open(stats_path, "w") as stats_fp:
                    stats[dataset_im_name][reconstruction_subdir] = summaries

                    json.dump(stats, stats_fp)
        
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
    
    parser.add_argument("--model_dir", "-M", required=True,
                        help="Model directory where we will save the checkpoints and Tensorboard logs")
    
    subparsers = parser.add_subparsers(title="mode",
                                       dest="mode",
                                       help="Current available modes: train")
    
    # ========================================================================
    # Training mode
    # ========================================================================
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
    
    train_subparsers = train_mode.add_subparsers(title="model",
                                                 dest="model",
                                                 help="Current available modes: vae, pln")
    
    train_subparsers.required = True
    
    # ========================================================================
    # Compression mode
    # ========================================================================
    compress_mode = subparsers.add_parser("compress",
                                           formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                           description="Compress using a trained model")
    
    compress_mode.add_argument("--input_file", required=True,
                               help="File to compress")
    compress_mode.add_argument("--output_file", required=True,
                               help="Output file")
    
    compress_mode.add_argument("--seed", default=42, type=int,
                            help="Seed to use in the compressor")
    compress_mode.add_argument("--n_steps", default=30, type=int,
                            help="Number of shards for the greedy sampler")
    compress_mode.add_argument("--n_bits_per_step", default=14, type=int,
                            help="Number of bits used to code a sample from one shard in the greedy sampler")
    compress_mode.add_argument("--first_level_max_group_size_bits", default=12, type=int,
                            help="Number of bits used to code group sizes in the first-level sampler")
    compress_mode.add_argument("--second_level_n_bits_per_group", default=20, type=int,
                            help="Maximum total group KL in bits in the second-level sampler")
    compress_mode.add_argument("--second_level_max_group_size_bits", default=4, type=int,
                            help="The number of bits used to code group sizes in the second-level sampler")
    compress_mode.add_argument("--second_level_dim_kl_bit_limit", default=12, type=int,
                            help="Maximum KL of a single dimension before it is deemed an outlier in the second-level sampler")
    compress_mode.add_argument("--outlier_index_bytes", default=2, type=int,
                            help="Bytes dedicated to coding the outliers' indices in the second-level sampler")
    compress_mode.add_argument("--outlier_sample_bytes", default=3, type=int,
                            help="Bytes dedicated to coding the outlier samples in the second-level sampler")

    compress_subparsers = compress_mode.add_subparsers(title="model",
                                                       dest="model",
                                                       help="Current available modes: vae, pln")
    compress_subparsers.required = True
    
    # ========================================================================
    # Decompress mode
    # ========================================================================
    decompress_mode = subparsers.add_parser("decompress",
                                            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                            description="Compress using a trained model")
    
    decompress_mode.add_argument("--comp_file", required=True,
                                 help="File to compress")
    decompress_mode.add_argument("--output_file", required=True,
                                 help="Output file")
    
    decompress_subparsers = decompress_mode.add_subparsers(title="model",
                                                           dest="model",
                                                           help="Current available modes: vae, pln")
    decompress_subparsers.required = True
    
    # ========================================================================
    # Compression statistics mode
    # ========================================================================
    compress_ds_mode = subparsers.add_parser("compress_ds",
                                           formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                           description="Compress a whole dataset using a trained model")
    
    compress_ds_mode.add_argument("--dataset_path", required=True,
                               help="Path to compress dataset to compress")
    compress_ds_mode.add_argument("--reconstruction_subdir", required=True,
                               help="Subdirectory to save files in")
    compress_ds_mode.add_argument("--reconstruction_root", required=False,
                                  default="/scratch/gf332/data/kodak_cwoq/",
                                  help="Reconstruction root directory, for grouping stuff together")
    compress_ds_mode.add_argument("--theoretical", action="store_true", default=False,
                                 help="Save stats for the theoretical optimal compression quality and size, or actual.")
    
    compress_ds_mode.add_argument("--seed", default=42, type=int,
                            help="Seed to use in the compressor")
    compress_ds_mode.add_argument("--n_steps", default=30, type=int,
                            help="Number of shards for the greedy sampler")
    compress_ds_mode.add_argument("--n_bits_per_step", default=14, type=int,
                            help="Number of bits used to code a sample from one shard in the greedy sampler")
    compress_ds_mode.add_argument("--first_level_max_group_size_bits", default=12, type=int,
                            help="Number of bits used to code group sizes in the first-level sampler")
    compress_ds_mode.add_argument("--second_level_n_bits_per_group", default=20, type=int,
                            help="Maximum total group KL in bits in the second-level sampler")
    compress_ds_mode.add_argument("--second_level_max_group_size_bits", default=4, type=int,
                            help="The number of bits used to code group sizes in the second-level sampler")
    compress_ds_mode.add_argument("--second_level_dim_kl_bit_limit", default=12, type=int,
                            help="Maximum KL of a single dimension before it is deemed an outlier in the second-level sampler")
    compress_ds_mode.add_argument("--outlier_index_bytes", default=2, type=int,
                            help="Bytes dedicated to coding the outliers' indices in the second-level sampler")
    compress_ds_mode.add_argument("--outlier_sample_bytes", default=3, type=int,
                            help="Bytes dedicated to coding the outlier samples in the second-level sampler")

    compress_ds_subparsers = compress_ds_mode.add_subparsers(title="model",
                                                           dest="model",
                                                           help="Current available modes: vae, pln")
    compress_ds_subparsers.required = True
    
    # ========================================================================
    # Add model specific stuff to each subparser
    # ========================================================================
    
    for subpars in [train_subparsers, compress_subparsers, decompress_subparsers, compress_ds_subparsers]:
        vae_model_parser = subpars.add_parser("vae",
                                                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                                description="Train a VAE")

        vae_model_parser.add_argument("--filters", default=128, type=int,
                                        help="Number of filters for the transforms")
        vae_model_parser.add_argument("--latent_channels", default=128, type=int,
                                        help="Number of channels in the latent space")
        # Train the PLN
        pln_model_parser = subpars.add_parser("pln",
                                                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                                description="Train a PLN")

        pln_model_parser.add_argument("--filters1", default=196, type=int,
                                        help="Number of filters for the first-level transforms")
        pln_model_parser.add_argument("--filters2", default=128, type=int,
                                        help="Number of filters for the second-level transforms")
        pln_model_parser.add_argument("--latent_channels1", default=128, type=int,
                                        help="Number of channels in the first-level latent space")
        pln_model_parser.add_argument("--latent_channels2", default=24, type=int,
                                        help="Number of channels in the second-level latent space")
    
    # Parse arguments
    args = parser.parse_args(argv[1:])
    if args.mode is None:
        parser.print_usage()
        sys.exit(2)
        
    return args


def main(args):
    if args.mode == "train":
        train(args)
    elif args.mode == "compress":
        compress(args)
    elif args.mode == "decompress":
        decompress(args)
    elif args.mode == "compress_ds":
        compress_dataset(args)

if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
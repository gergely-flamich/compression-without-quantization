import argparse
import glob
import os
import json
import time

from tqdm import tqdm

from absl import app
from absl.flags import argparse_flags

from miracle_arguments import parse_args

from pln import ProbabilisticLadderNetwork
from vae import VariationalAutoEncoder

import tensorflow.compat.v1 as tf

import numpy as np

import matplotlib.pyplot as plt

# empirical distribution postfixes
first_level_group_size_postfix = "_1.npy"
second_level_group_size_postfix = "_2.npy"

first_level_index_postfix = "_samp_ind_1.npy"
second_level_index_postfix = "_samp_ind_2.npy"

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
        
#         model_dir = "/scratch/gf332/data/kodak_cwoq_distributions/"
        
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
                                greedy_max_group_size_bits=args.greedy_max_group_size_bits,
                                
                                comp_file_path=args.output_file,
                                
                                use_importance_sampling=args.use_importance_sampling,
                                
                                second_level_n_bits_per_group=args.second_level_n_bits_per_group,
                                second_level_max_group_size_bits=args.second_level_max_group_size_bits,
                                second_level_dim_kl_bit_limit=args.second_level_dim_kl_bit_limit,
                                
                                first_level_n_bits_per_group=args.first_level_n_bits_per_group,
                                first_level_max_group_size_bits=args.first_level_max_group_size_bits,
                                first_level_dim_kl_bit_limit=args.first_level_dim_kl_bit_limit,
                                
                                outlier_index_bytes=args.outlier_index_bytes,
                                outlier_sample_bytes=args.outlier_sample_bytes,
                                
                                use_index_ac=args.use_index_ac,
                
                                first_level_group_dist_counts=args.dist_prefix + first_level_group_size_postfix,
                                second_level_group_dist_counts=args.dist_prefix + second_level_group_size_postfix,

                                first_level_sample_index_counts=args.dist_prefix + first_level_index_postfix,
                                second_level_sample_index_counts=args.dist_prefix + second_level_index_postfix,
                                
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
                                                   use_importance_sampling=args.use_importance_sampling,
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
                                                        greedy_max_group_size_bits=args.greedy_max_group_size_bits,
                                                        comp_file_path=comp_file_paths[i],
                                                           
                                                        use_importance_sampling=args.use_importance_sampling,
                                                           
                                                        second_level_n_bits_per_group=args.second_level_n_bits_per_group,
                                                        second_level_max_group_size_bits=args.second_level_max_group_size_bits,
                                                        second_level_dim_kl_bit_limit=args.second_level_dim_kl_bit_limit,
                                                           
                                                        first_level_n_bits_per_group=args.first_level_n_bits_per_group,
                                                        first_level_max_group_size_bits=args.first_level_max_group_size_bits,
                                                        first_level_dim_kl_bit_limit=args.first_level_dim_kl_bit_limit,
                                                           
                                                        outlier_index_bytes=args.outlier_index_bytes,
                                                        outlier_sample_bytes=args.outlier_sample_bytes,
                                                           
                                                        use_index_ac=args.use_index_ac,
                
                                                        first_level_group_dist_counts=args.dist_prefix + first_level_group_size_postfix,
                                                        second_level_group_dist_counts=args.dist_prefix + second_level_group_size_postfix,

                                                        first_level_sample_index_counts=args.dist_prefix + first_level_index_postfix,
                                                        second_level_sample_index_counts=args.dist_prefix + second_level_index_postfix,
                                                           
                                                        verbose=args.verbose)

                    encoding_time = time.time() - start_time
                    
                    summaries["image_shape"] = summaries["image_shape"].tolist()

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
# Build empirical coding distribution for importance group samples
# ============================================================================
# ============================================================================     
def build_empirical_dists(args,
                          image_regex="*.png"):
    
    dataset_im_paths = glob.glob(args.data_path + "/" + image_regex)
    
    paths_ds = tf.data.Dataset.from_tensor_slices(dataset_im_paths)
    image_ds = paths_ds.map(
        read_png, num_parallel_calls=16)
    image_ds = image_ds.prefetch(32)
    
    image = image_ds.make_one_shot_iterator().get_next()
    next_image = tf.expand_dims(image, 0)
    next_image.set_shape([1, None, None, 3])
    
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
        
        # The +1 is for the EOF symbol
        group_sizes2 = np.zeros(1 + 2**args.second_level_max_group_size_bits, dtype=np.int64)
        group_sizes1 = np.zeros(1 + 2**args.first_level_max_group_size_bits, dtype=np.int64)
        
        # The +1 is for the EOF symbol
        sample_indices2 = np.zeros(1 + 2**args.second_level_n_bits_per_group, dtype=np.int64)
        sample_indices1 = np.zeros(1 + 2**args.first_level_n_bits_per_group, dtype=np.int64)
        
        for i in range(len(dataset_im_paths)):
            
            print("=====================================")
            print("Calculating statistics for image {}".format(i))
            print("=====================================")
            image = sess.run(next_image)
            
            greedy_coder = lambda gs_1, gs_2, ind_1, ind_2: model.code_image_greedy(
                session=sess,
                image=image, 

                seed=args.seed, 

                n_steps=args.n_steps,
                n_bits_per_step=args.n_bits_per_step,
                greedy_max_group_size_bits=args.greedy_max_group_size_bits,

                comp_file_path=None,

                use_importance_sampling=True,

                second_level_n_bits_per_group=args.second_level_n_bits_per_group,
                second_level_max_group_size_bits=args.second_level_max_group_size_bits,
                second_level_dim_kl_bit_limit=args.second_level_dim_kl_bit_limit,

                first_level_n_bits_per_group=args.first_level_n_bits_per_group,
                first_level_max_group_size_bits=args.first_level_max_group_size_bits,
                first_level_dim_kl_bit_limit=args.first_level_dim_kl_bit_limit,

                outlier_index_bytes=args.outlier_index_bytes,
                outlier_sample_bytes=args.outlier_sample_bytes,

                return_first_level_group_sizes=gs_1,
                return_second_level_group_sizes=gs_2,
                
                return_first_level_indices=ind_1,
                return_second_level_indices=ind_2,
                
                use_index_ac=args.use_index_ac,
                
                first_level_group_dist_counts=args.dist_prefix + first_level_group_size_postfix,
                second_level_group_dist_counts=args.dist_prefix + second_level_group_size_postfix,
                
                first_level_sample_index_counts=args.dist_prefix + first_level_index_postfix,
                second_level_sample_index_counts=args.dist_prefix + second_level_index_postfix,

                verbose=args.verbose)
            
            # ====================================================================
            # Group indices
            # ====================================================================

            # Get the group indices on the second level
            gi2 = greedy_coder(False, True, False, False)
            
            group_differences2 = gi2[1:] - gi2[:-1]   
            unique, counts = np.unique(group_differences2, return_counts=True)
            
            group_sizes2[unique] += counts
            group_sizes2[0] += 1
            
            # Get the group indices on the first level
            gi1 = greedy_coder(True, False, False, False)
            
            group_differences1 = gi1[1:] - gi1[:-1]
            
            unique, counts = np.unique(group_differences1, return_counts=True)
            
            group_sizes1[unique] += counts
            group_sizes1[0] += 1
            
            np.save(args.output_path + "_2.npy", group_sizes2)
            np.save(args.output_path + "_1.npy", group_sizes1)
            
            # ====================================================================
            # Sample indices
            # ====================================================================
            
            # Get the sample indices on the second level
            ind_2 = greedy_coder(False, False, False, True)
            unique, counts = np.unique(ind_2, return_counts=True)
            
            # Shift everything by one
            sample_indices2[unique + 1] += counts
            sample_indices2[0] += 1
            
             # Get the sample indices on the first level
            ind_1 = greedy_coder(False, False, True, False)
            unique, counts = np.unique(ind_1, return_counts=True)
            
            # Shift everything by one
            sample_indices1[unique + 1] += counts
            sample_indices1[0] += 1
            
            np.save(args.output_path + "_samp_ind_2.npy", sample_indices2)
            np.save(args.output_path + "_samp_ind_1.npy", sample_indices1)
            
        
def main(args):
    if args.mode == "train":
        train(args)
    elif args.mode == "compress":
        compress(args)
    elif args.mode == "decompress":
        decompress(args)
    elif args.mode == "compress_ds":
        compress_dataset(args)
    elif args.mode == "build_dists":
        build_empirical_dists(args)

if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
import argparse

from absl import app
from absl.flags import argparse_flags

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
    
    train_mode.add_argument("--data_path", "-D", required=True,
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
    
    decompress_mode.add_argument("--use_importance_sampling", default=True, action="store_true",
                                    help="Should we use importance sampling on the first level?")
    decompress_mode.add_argument("--use_index_ac", default=False, action="store_true",
                            help="Should we use arithmetic coding for the MIRACLE indices?")
    decompress_mode.add_argument("--dist_prefix", required=True,
                            help="Path to the empirical distributions for group size and index coding.")
            
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
    

    compress_ds_subparsers = compress_ds_mode.add_subparsers(title="model",
                                                           dest="model",
                                                           help="Current available modes: vae, pln")
    compress_ds_subparsers.required = True
    
    # ========================================================================
    # Compression statistics mode
    # ========================================================================
    build_dists_mode = subparsers.add_parser("build_dists",
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                       description="Train a new model")
    
    build_dists_mode.add_argument("--data_path", "-D", required=True,
                                help="Path to PNG dataset")
    build_dists_mode.add_argument("--output_path", "-O", required=True,
                                help="Where to put the pickled numpy array of probabilities.")
    
    build_dists_mode.add_argument("--preprocess_threads", default=16, type=int,
                                help="Number of threads to use to process the dataset")
    
    build_dists_subparsers = build_dists_mode.add_subparsers(title="model",
                                                            dest="model",
                                                            help="Current available modes: vae, pln")
     
    # ========================================================================
    # Add compression specific stuff to subparser modes
    # ========================================================================
    for mode in [compress_mode, compress_ds_mode, build_dists_mode]:
        
            mode.add_argument("--seed", default=42, type=int,
                            help="Seed to use in the compressor")
            
            mode.add_argument("--use_importance_sampling", default=False, action="store_true",
                            help="Should we use importance sampling on the first level?")
            
            mode.add_argument("--use_index_ac", default=False, action="store_true",
                            help="Should we use arithmetic coding for the MIRACLE indices?")
            
            mode.add_argument("--dist_prefix", required=True,
                            help="Path to the empirical distributions for group size and index coding.")
            
            mode.add_argument("--n_steps", default=30, type=int,
                            help="Number of shards for the greedy sampler")
            
            mode.add_argument("--n_bits_per_step", default=14, type=int,
                            help="Number of bits used to code a sample from one shard in the greedy sampler")
            
            mode.add_argument("--greedy_max_group_size_bits", default=12, type=int,
                            help="Number of bits used to code group sizes in the first-level greedy sampler")
            
            mode.add_argument("--second_level_n_bits_per_group", default=20, type=int,
                            help="Maximum total group KL in bits in the second-level sampler")
            
            mode.add_argument("--second_level_max_group_size_bits", default=2, type=int,
                            help="The number of bits used to code group sizes in the second-level sampler")
            
            mode.add_argument("--second_level_dim_kl_bit_limit", default=16, type=int,
                            help="Maximum KL of a single dimension before it is deemed an outlier in the second-level sampler")
            
            mode.add_argument("--first_level_n_bits_per_group", default=20, type=int,
                            help="Maximum total group KL in bits in the first-level sampler")
            
            mode.add_argument("--first_level_max_group_size_bits", default=4, type=int,
                            help="The number of bits used to code group sizes in the first-level sampler")
            
            mode.add_argument("--first_level_dim_kl_bit_limit", default=16, type=int,
                            help="Maximum KL of a single dimension before it is deemed an outlier in the first-level sampler")
            
            mode.add_argument("--outlier_index_bytes", default=3, type=int,
                            help="Bytes dedicated to coding the outliers' indices in the second-level sampler")
            
            mode.add_argument("--outlier_sample_bytes", default=2, type=int,
                            help="Bytes dedicated to coding the outlier samples in the second-level sampler")
        
    # ========================================================================
    # Add model specific stuff to each subparser
    # ========================================================================
    
    for subpars in [train_subparsers, 
                    compress_subparsers, 
                    decompress_subparsers, 
                    compress_ds_subparsers, 
                    build_dists_subparsers]:
        
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
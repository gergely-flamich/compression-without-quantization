import os

import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras
tfl = tf.keras.layers

from losses import SsimMultiscalePseudo, MixPseudo

from transforms import AnalysisTransform_1, AnalysisTransform_2, SynthesisTransform_1, SynthesisTransform_2

from coded_greedy_sampler import code_grouped_greedy_sample, decode_grouped_greedy_sample
from coded_importance_sampler import code_grouped_importance_sample, decode_grouped_importance_sample

from binary_io import write_bin_code, read_bin_code

# =====================================================================
# =====================================================================
# PLN definition
# =====================================================================
# =====================================================================

class ProbabilisticLadderNetwork(tfk.Model):
    
    def __init__(self,
                 first_level_filters,
                 second_level_filters,
                 first_level_latent_channels,
                 second_level_latent_channels,
                 padding="same_zeros",
                 likelihood="gaussian",
                 learn_gamma=False,
                 *args,
                 **kwargs):
        
        self.first_level_filters = first_level_filters
        self.second_level_filters = second_level_filters
        self.first_level_latent_channels = first_level_latent_channels
        self.second_level_latent_channels = second_level_latent_channels
        self.padding = padding
        
        self.learn_gamma = learn_gamma
        
        if likelihood == "gaussian":
            self.likelihood_dist_family = tfd.Normal
            
        elif likelihood == "laplace":
            self.likelihood_dist_family = tfd.Laplace
            
        elif likelihood == "ssim":
            self.likelihood_dist_family = SsimMultiscalePseudo
            
        elif likelihood == "mix":
            self.likelihood_dist_family = MixPseudo
            
        else:
            raise Exception("Unknown likelihood: {}".format(likelihood))
        
        super(ProbabilisticLadderNetwork, self).__init__(*args, **kwargs)
        
    @property
    def first_level_kl(self):
        return tfd.kl_divergence(self.posterior_1, self.prior_1)
        
        
    @property
    def second_level_kl(self):
        return tfd.kl_divergence(self.posterior_2, self.prior_2)    
        
        
    def bpp(self, num_pixels):
        return (tf.reduce_sum(self.first_level_kl) +
                tf.reduce_sum(self.second_level_kl)) / (np.log(2) * num_pixels)
    
    def save_latent_statistics(self, sess, dir_path):
        
        q1_loc = self.posterior_1.loc.eval(session = sess)
        q1_scale = self.posterior_1.scale.eval(session = sess)

        p1_loc = self.prior_1.loc.eval(session = sess)
        p1_scale = self.prior_1.scale.eval(session = sess)
        
        q2_loc = self.posterior_2.loc.eval(session = sess)
        q2_scale = self.posterior_2.scale.eval(session = sess)
        
        p2_loc = self.prior_2.loc.eval(session = sess)
        p2_scale = self.prior_2.scale.eval(session = sess)
        
        np.save(dir_path + "/pln_q1_loc.npy", q1_loc)
        np.save(dir_path + "/pln_q1_scale.npy", q1_scale)
        np.save(dir_path + "/pln_p1_loc.npy", p1_loc)
        np.save(dir_path + "/pln_p1_scale.npy", p1_scale)
        np.save(dir_path + "/pln_q2_loc.npy", q2_loc)
        np.save(dir_path + "/pln_q2_scale.npy", q2_scale)
        np.save(dir_path + "/pln_p2_loc.npy", p2_loc)
        np.save(dir_path + "/pln_p2_scale.npy", p2_scale)
        
    def build(self, input_shape):
        
        self.analysis_transform_1 = AnalysisTransform_1(num_filters=self.first_level_filters,
                                                        num_latent_channels=self.first_level_latent_channels,
                                                        padding=self.padding)
        
        self.synthesis_transform_1 = SynthesisTransform_1(num_filters=self.first_level_filters,
                                                          padding=self.padding)
        
        self.analysis_transform_2 = AnalysisTransform_2(num_filters=self.second_level_filters,
                                                        num_latent_channels=self.second_level_latent_channels,
                                                        padding=self.padding)
        
        self.synthesis_transform_2 = SynthesisTransform_2(num_filters=self.second_level_filters,
                                                          num_output_channels=self.first_level_latent_channels,
                                                          padding=self.padding)
        
        if self.learn_gamma:
            self.log_gamma = self.add_variable("log_gamma", 
                                               dtype=tf.float32, 
                                               initializer=tf.compat.v1.constant_initializer(0.),
                                               trainable=True)
            
        else:
            self.log_gamma = tf.constant(0., dtype=tf.float32)
        
        super(ProbabilisticLadderNetwork, self).build(input_shape)
        
    
    def call(self, inputs, eps=1e-12):
        
        # Perform first analysis transform
        self.analysis_transform_1(inputs)
        
        # Pass on the predicted mean to the second analysis transform
        # and get the sample from the second level
        z_2 = self.analysis_transform_2(self.analysis_transform_1.loc)
        
        # Get the first-level priors by performing the second level synthesis transform
        # on the second-level sample
        self.synthesis_transform_2(z_2)
        
        # To get the first-level posteriors, combine the first-level likelihoods
        # and the first-level priors appropriately
        likelihood_loc = self.analysis_transform_1.loc
        prior_loc = self.synthesis_transform_2.loc
        
        likelihood_var = tf.square(self.analysis_transform_1.scale)
        prior_var = tf.square(self.synthesis_transform_2.scale)

        likelihood_prec = 1. / (likelihood_var + eps)
        prior_prec = 1. / (prior_var + eps)

        # Combined variance
        combined_var = 1. / (likelihood_prec + prior_prec)
        combined_scale = tf.sqrt(combined_var)

        # Combined location
        combined_loc = likelihood_loc * prior_prec
        combined_loc += prior_loc * likelihood_prec
        combined_loc *= combined_var
        
        # Set latent distributions
        self.posterior_1 = tfd.Normal(loc=combined_loc,
                                      scale=combined_scale)
        
        self.prior_1 = self.synthesis_transform_2.prior
        
        self.posterior_2 = self.analysis_transform_2.posterior
        self.prior_2 = self.analysis_transform_2.prior
        
        # Get first level sample
        z_1 = self.posterior_1.sample()
        
        # Perform first level synthesis transform on the first level sample
        reconstruction = self.synthesis_transform_1(z_1)
        
        self.likelihood_dist = self.likelihood_dist_family(loc=reconstruction,
                                                           scale=tf.math.exp(self.log_gamma))
        
        self.log_likelihood = self.likelihood_dist.log_prob(inputs)
        
        return reconstruction
    
    # =================================================================
    # =================================================================
    #
    # Greedy Compression
    #
    # =================================================================
    # =================================================================
    
    def code_image_greedy(self,
                          session,
                          image, 
                          seed, 
                          n_steps,
                          n_bits_per_step,
                          comp_file_path,
                          backfitting_steps_level_1=0,
                          backfitting_steps_level_2=0,
                          use_log_prob=False,
                          rho=1.,
                          use_importance_sampling=True,
                          first_level_max_group_size_bits=12,
                          second_level_n_bits_per_group=20,
                          second_level_max_group_size_bits=4,
                          second_level_dim_kl_bit_limit=12,
                          outlier_index_bytes=3,
                          outlier_sample_bytes=2,
                          verbose=False):
        
        # -------------------------------------------------------------------------------------
        # Step 1: Set the latent distributions for the image
        # -------------------------------------------------------------------------------------
        
        if verbose: print("Calculating latent distributions for image...")
        session.run(self.call(image))
        
        image_shape, first_level_shape, second_level_shape = session.run([tf.shape(image),
                                                                          tf.shape(self.posterior_1.loc), 
                                                                          tf.shape(self.posterior_2.loc)])
        
        # -------------------------------------------------------------------------------------
        # Step 2: Create a coded sample of the latent space
        # -------------------------------------------------------------------------------------
        
        if verbose: print("Coding second level...")
            
        if use_importance_sampling:
            
            sample2, code2, group_indices2, outlier_extras2 = code_grouped_importance_sample(
                sess=session,
                target=self.posterior_2,
                proposal=self.prior_2, 
                n_bits_per_group=second_level_n_bits_per_group, 
                seed=seed, 
                max_group_size_bits=second_level_max_group_size_bits,
                dim_kl_bit_limit=second_level_dim_kl_bit_limit)
            
            outlier_extras2 = list(map(lambda x: x.reshape([-1]), outlier_extras2))
            
        else:
            sample2, code2, group_indices2 = code_grouped_greedy_sample(sess=session,
                                                                        target=self.posterior_2, 
                                                                        proposal=self.prior_2, 
                                                                        n_bits_per_step=n_bits_per_step, 
                                                                        n_steps=n_steps, 
                                                                        seed=seed, 
                                                                        max_group_size_bits=second_level_max_group_size_bits,
                                                                        adaptive=True,
                                                                        backfitting_steps=backfitting_steps_level_2,
                                                                        use_log_prob=use_log_prob,
                                                                        rho=rho)
            
        # We will encode the group differences as this will cost us less
        group_differences2 = [0]
        
        for i in range(1, len(group_indices2)):
            group_differences2.append(group_indices2[i] - group_indices2[i - 1])
        
        # We need to adjust the priors to the second stage sample
        latents = tf.reshape(sample2, second_level_shape)
        session.run(self.synthesis_transform_2(latents))
        
        self.prior_1 = self.synthesis_transform_2.prior
        
        if verbose: print("Coding first level")
            
        sample1, code1, group_indices1 = code_grouped_greedy_sample(sess=session,
                                                                    target=self.posterior_1, 
                                                                    proposal=self.prior_1, 
                                                                    n_bits_per_step=n_bits_per_step, 
                                                                    n_steps=n_steps, 
                                                                    seed=seed, 
                                                                    max_group_size_bits=first_level_max_group_size_bits,
                                                                    backfitting_steps=backfitting_steps_level_1,
                                                                    use_log_prob=use_log_prob,
                                                                    adaptive=True)
        
        # We will encode the group differences as this will cost us less
        group_differences1 = [0]
        
        for i in range(1, len(group_indices1)):
            group_differences1.append(group_indices1[i] - group_indices1[i - 1])
        
        
        bitcode = code1 + code2.decode("utf-8")
        # -------------------------------------------------------------------------------------
        # Step 3: Write the compressed file
        # -------------------------------------------------------------------------------------
        
        extras = [seed, n_steps, n_bits_per_step] + \
                 first_level_shape[1:3].tolist() + \
                 second_level_shape[1:3].tolist()
        
        var_length_extras = [group_differences1, group_differences2]
        var_length_bits = [first_level_max_group_size_bits,  
                           second_level_max_group_size_bits]
        
        if use_importance_sampling:
            
            var_length_extras += outlier_extras2
            var_length_bits += [ outlier_index_bytes * 8, outlier_sample_bytes * 8 ]
    
        write_bin_code(bitcode, 
                       comp_file_path, 
                       extras=extras,
                       var_length_extras=var_length_extras,
                       var_length_bits=var_length_bits)
        
        # -------------------------------------------------------------------------------------
        # Step 4: Some logging information
        # -------------------------------------------------------------------------------------
        
        total_kls = [tf.reduce_sum(x) for x in (self.first_level_kl, self.second_level_kl)]
        total_kls = session.run(total_kls)
        total_kl = sum(total_kls)

        theoretical_byte_size = (total_kl + 2 * np.log(total_kl + 1)) / np.log(2) / 8
        extra_byte_size = len(group_indices1) * var_length_bits[0] // 8 + \
                          len(group_indices2) * var_length_bits[1] // 8 + 7 * 2
        actual_byte_size = os.path.getsize(comp_file_path)

        actual_no_extra = actual_byte_size - extra_byte_size
        
        first_level_theoretical = (total_kls[0] + 2 * np.log(total_kls[0] + 1)) / np.log(2) / 8
        first_level_actual_no_extra = len(code1) / 8
        first_level_extra = len(group_indices1) * var_length_bits[0] // 8

        sample1_reshaped = tf.reshape(sample1, first_level_shape)
        first_level_avg_log_lik = tf.reduce_mean(self.posterior_1.log_prob(sample1_reshaped))
        first_level_sample_avg = tf.reduce_mean(self.posterior_1.log_prob(self.posterior_1.sample()))
        
        first_level_avg_log_lik, first_level_sample_avg = session.run([first_level_avg_log_lik, first_level_sample_avg])
        
        
        second_level_theoretical = (total_kls[1] + 2 * np.log(total_kls[1] + 1)) / np.log(2) / 8
        second_level_actual_no_extra = len(code2) / 8
        second_level_extra = len(group_indices2) * var_length_bits[1] // 8 + 1
        
        second_bpp = (second_level_actual_no_extra + second_level_extra) * 8 / (image_shape[1] * image_shape[2]) 

        sample2_reshaped = tf.reshape(sample2, second_level_shape)
        second_level_avg_log_lik = tf.reduce_mean(self.posterior_2.log_prob(sample2_reshaped))
        second_level_sample_avg = tf.reduce_mean(self.posterior_2.log_prob(self.posterior_2.sample()))
        
        second_level_avg_log_lik, second_level_sample_avg = session.run([second_level_avg_log_lik, second_level_sample_avg])
        
        bpp = 8 * actual_byte_size / (image_shape[1] * image_shape[2]) 
        
        summaries = {
            "image_shape": image_shape,
            "theoretical_byte_size": float(theoretical_byte_size),
            "actual_byte_size": actual_byte_size,
            "extra_byte_size": extra_byte_size,
            "actual_no_extra": actual_no_extra,
            "second_bpp": second_bpp,
            "bpp": bpp
        }
        
        if verbose:

            print("Image dimensions: {}".format(image_shape))
            print("Theoretical size: {:.2f} bytes".format(theoretical_byte_size))
            print("Actual size: {:.2f} bytes".format(actual_byte_size))
            print("Extra information size: {:.2f} bytes {:.2f}% of actual size".format(extra_byte_size, 
                                                                                       100 * extra_byte_size / actual_byte_size))
            print("Actual size without extras: {:.2f} bytes".format(actual_no_extra))
            print("Efficiency: {:.3f}".format(actual_byte_size / theoretical_byte_size))
            print("")
            
            print("First level theoretical size: {:.2f} bytes".format(first_level_theoretical))
            print("First level actual (no extras) size: {:.2f} bytes".format(first_level_actual_no_extra))
            print("First level extras size: {:.2f} bytes".format(first_level_extra))
            print("First level Efficiency: {:.3f}".format(
                (first_level_actual_no_extra + first_level_extra) / first_level_theoretical))
            
            print("First level # of groups: {}".format(len(group_indices1)))
            print("First level greedy sample average log likelihood: {:.4f}".format(first_level_avg_log_lik))
            print("First level average sample log likelihood on level 1: {:.4f}".format(first_level_sample_avg))
            print("")
           
            print("Second level theoretical size: {:.2f} bytes".format(second_level_theoretical))
            print("Second level actual (no extras) size: {:.2f} bytes".format(second_level_actual_no_extra))
            print("Second level extras size: {:.2f} bytes".format(second_level_extra))

            if use_importance_sampling:
                print("{} outliers were not compressed (higher than {} bits of KL)".format(len(outlier_extras2[0]),
                                                                                           second_level_dim_kl_bit_limit))
            print("Second level Efficiency: {:.3f}".format(
                (second_level_actual_no_extra + second_level_extra) / second_level_theoretical))
            print("Second level's contribution to bpp: {:.4f}".format(second_bpp))
            print("Second level # of groups: {}".format(len(group_indices2)))
            print("Second level importance sample average log likelihood: {:.4f}".format(second_level_avg_log_lik))
            print("Second level average sample log likelihood on level 1: {:.4f}".format(second_level_sample_avg))
            print("")
            
            print("{:.4f} bits / pixel".format( bpp ))
        
        return (sample2, sample1), summaries
    
    
    # =================================================================
    # =================================================================
    #
    # Greedy Decompression
    #
    # =================================================================
    # =================================================================
    
    def decode_image_greedy(self,
                            session,
                            comp_file_path,
                            use_importance_sampling=True,
                            rho=1.,
                            verbose=False):
        
        # -------------------------------------------------------------------------------------
        # Step 1: Read the compressed file
        # -------------------------------------------------------------------------------------
        
        # the extras are: seed, n_steps, n_bits_per_step and W x H of the two latent levels
        # var length extras are the two lists of group indices
        num_var_length_extras = 2
        
        if use_importance_sampling:
            num_var_length_extras += 2
        
        code, extras, var_length_extras = read_bin_code(comp_file_path, 
                                                        num_extras=7, 
                                                        num_var_length_extras=num_var_length_extras)
        
        seed = extras[0]
        
        n_steps = extras[1]
        n_bits_per_step = extras[2]
        
        # Get shape information back
        first_level_shape = [1] + extras[3:5] + [self.first_level_latent_channels]
        second_level_shape = [1] + extras[5:] + [self.second_level_latent_channels]
        
        # Total number of latents on levels
        num_first_level = np.prod(first_level_shape)
        num_second_level = np.prod(second_level_shape)
        
        first_code_length = n_steps * n_bits_per_step * (len(var_length_extras[0]) - 1)
        second_code_length = n_steps * n_bits_per_step * (len(var_length_extras[1]) - 1)
        
        code1 = code[:first_code_length]
        code2 = code[first_code_length:first_code_length + second_code_length]
        
        # -------------------------------------------------------------------------------------
        # Step 2: Decode the samples
        # -------------------------------------------------------------------------------------
        
        # Decode second level
        proposal = tfd.Normal(loc=tf.zeros(second_level_shape),
                              scale=tf.ones(second_level_shape))
        
        
        # Get group indices back
        group_differences2 = var_length_extras[1]
        
        group_indices2 = [0]
        
        for i in range(1, len(group_differences2)):
            group_indices2.append(group_indices2[i - 1] + group_differences2[i])
        
        print("Decoding second level")
        if use_importance_sampling:
            decoded_second_level = decode_grouped_importance_sample(sess=session,
                                                                    bitcode=code2, 
                                                                    group_start_indices=group_indices2[:-1],
                                                                    proposal=proposal, 
                                                                    n_bits_per_group=20,
                                                                    seed=seed,
                                                                    outlier_indices=var_length_extras[2],
                                                                    outlier_samples=var_length_extras[3])
        
        else:
            decoded_second_level = decode_grouped_greedy_sample(sess=session,
                                                                bitcode=code2, 
                                                                group_start_indices=var_length_extras[1],
                                                                proposal=proposal, 
                                                                n_bits_per_step=n_bits_per_step, 
                                                                n_steps=n_steps, 
                                                                seed=seed,
                                                                rho=rho,
                                                                adaptive=True)
        
        decoded_second_level = tf.reshape(decoded_second_level, second_level_shape)
        
        # Now we can calculate the the first level priors
        session.run(self.synthesis_transform_2(decoded_second_level))
        self.prior_1 = self.synthesis_transform_2.prior
        
        # Get group indices back
        group_differences1 = var_length_extras[0]
        
        group_indices1 = [0]
        
        for i in range(1, len(group_differences1)):
            group_indices1.append(group_indices1[i - 1] + group_differences1[i])
        
        # Decode first level
        print("Decoding first level")
        decoded_first_level = decode_grouped_greedy_sample(sess=session,
                                                           bitcode=code1, 
                                                            group_start_indices=group_indices1,
                                                            proposal=self.prior_1, 
                                                            n_bits_per_step=n_bits_per_step, 
                                                            n_steps=n_steps, 
                                                            seed=seed,
                                                            rho=rho,
                                                            adaptive=True)
        
        decoded_first_level = tf.reshape(decoded_first_level, first_level_shape)
        
        
        # -------------------------------------------------------------------------------------
        # Step 4: Reconstruct the image with the VAE
        # -------------------------------------------------------------------------------------
        
        reconstruction = self.synthesis_transform_1(decoded_first_level)
        
        return session.run(tf.squeeze(reconstruction))
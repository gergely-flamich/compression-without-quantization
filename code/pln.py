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

from coding import ArithmeticCoder

from binary_io import write_bin_code, read_bin_code

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
                          
                          # Greedy sampling parameters
                          n_steps,
                          n_bits_per_step,
                          greedy_max_group_size_bits,
                          
                          comp_file_path,
                          
                          backfitting_steps_level_1=0,
                          backfitting_steps_level_2=0,
                          use_log_prob=False,
                          rho=1.,
                          use_importance_sampling=False,
                          
                          # Importance sampling parameters
                          second_level_n_bits_per_group=20,
                          second_level_max_group_size_bits=4,
                          second_level_dim_kl_bit_limit=12,
                          first_level_n_bits_per_group=20,
                          first_level_max_group_size_bits=3,
                          first_level_dim_kl_bit_limit=12,
                          outlier_index_bytes=3,
                          outlier_sample_bytes=2,
                          
                          second_level_counts="/homes/gf332/compression-without-quantization/group_dists_2.npy",
                          first_level_counts="/homes/gf332/compression-without-quantization/group_dists_1.npy",
                          
                          return_first_level_group_sizes=False,
                          return_second_level_group_sizes=False,
                          
                          verbose=False):
        
        # -------------------------------------------------------------------------------------
        # Step 1: Set the latent distributions for the image
        # -------------------------------------------------------------------------------------
        
        if verbose: print("Calculating latent distributions for image...")
            
        ops = [self.call(image), 
               self.posterior_1.loc, 
               self.posterior_1.scale,
               self.posterior_2.loc, 
               self.posterior_2.scale,
               self.prior_2.loc, 
               self.prior_2.scale]
        
        im, q1_loc, q1_scale, q2_loc, q2_scale, p2_loc, p2_scale = session.run(ops)

        q1 = tfd.Normal(loc=q1_loc, scale=q1_scale)
        q2 = tfd.Normal(loc=q2_loc, scale=q2_scale)
        p2 = tfd.Normal(loc=p2_loc, scale=p2_scale)
        
        image_shape = list(im.shape)
        first_level_shape = list(q1_loc.shape)
        second_level_shape = list(q2_loc.shape)

        # -------------------------------------------------------------------------------------
        # Step 2: Create a coded sample of the latent space
        # -------------------------------------------------------------------------------------
        
        if verbose: print("Coding second level...")
           
        res = code_grouped_importance_sample(sess=session,
                                            target=q2,
                                            proposal=p2, 
                                            n_bits_per_group=second_level_n_bits_per_group, 
                                            seed=seed, 
                                            max_group_size_bits=second_level_max_group_size_bits,
                                            dim_kl_bit_limit=second_level_dim_kl_bit_limit,
                                            return_group_indices_only=return_second_level_group_sizes)
        
        if return_second_level_group_sizes:
            group_start_indices, group_kls = res
            
            return group_start_indices
            
        sample2, code2, group_indices2, outlier_extras2 = res
        np.save("/homes/gf332/compression-without-quantization/s2_cod.npy", sample2)
        
        
        outlier_extras2 = list(map(lambda x: x.reshape([-1]), outlier_extras2))
        
        # The -1 at the end shifts the range of the group sizes to the 0 - 15 range from the 1 - 16 range
        # such that it can be coded as usual
        
        group_differences2 = group_indices2[1:] - group_indices2[:-1]
        
        group_indices2_ = [0]
        
        for i in range(0, len(group_differences2)):
            # The +1 at the end shifts the group sizes back from the range 0 - 15 to the range 1 - 16.
            group_indices2_.append(group_indices2_[i] + (group_differences2[i] + 1))
         
        # We need to adjust the priors to the second stage sample
        latents = tf.reshape(sample2, second_level_shape)
        
        ops = [self.synthesis_transform_2(latents), 
               self.synthesis_transform_2.loc, 
               self.synthesis_transform_2.scale]
        
        _, p1_loc, p1_scale = session.run(ops)
        
        p1 = tfd.Normal(loc=p1_loc, scale=p1_scale)
        
        if verbose: print("Coding first level")
        
        if use_importance_sampling:
            
            group_index_counts = np.zeros(2**first_level_n_bits_per_group)
            
            res = code_grouped_importance_sample(sess=session,
                                                target=q1,
                                                proposal=p1, 
                                                n_bits_per_group=first_level_n_bits_per_group, 
                                                seed=seed, 
                                                max_group_size_bits=first_level_max_group_size_bits,
                                                dim_kl_bit_limit=first_level_dim_kl_bit_limit,
                                                return_group_indices_only=return_first_level_group_sizes)
            
            if return_first_level_group_sizes:
                group_start_indices, group_kls = res

                return group_start_indices
            
            sample1, code1, group_indices1, outlier_extras1 = res
            
            unique, counts = np.unique(group_indices1, return_counts=True)
            
            group_index_counts[unique] += counts
            
            np.save("gic.npy", group_index_counts)
            
            print(group_indices1[:30])
            print(group_indices1[-30:])
            print(code1[:70])
            print(code1[-70:])
            print(first_level_n_bits_per_group)
            
        else:
            sample1, code1, group_indices1 = code_grouped_greedy_sample(sess=session,
                                                                        target=q1, 
                                                                        proposal=p1, 
                                                                        n_bits_per_step=n_bits_per_step, 
                                                                        n_steps=n_steps, 
                                                                        seed=seed, 
                                                                        max_group_size_bits=greedy_max_group_size_bits,
                                                                        backfitting_steps=backfitting_steps_level_1,
                                                                        use_log_prob=use_log_prob,
                                                                        adaptive=True)

#         np.save("/homes/gf332/compression-without-quantization/s1_cod.npy", sample1)
                
        # We will encode the group differences as this will cost us less
        group_differences1 = group_indices1[1:] - group_indices1[:-1] 

        bitcode = code1.decode("utf-8") if use_importance_sampling else code1
        bitcode += code2.decode("utf-8")
        
        print(len(code1))
        print(len(code2))
        # -------------------------------------------------------------------------------------
        # Step 3: Write the compressed file
        # -------------------------------------------------------------------------------------
        
        extras = [seed, n_steps, n_bits_per_step, first_level_n_bits_per_group, second_level_n_bits_per_group] + \
                 first_level_shape[1:3] + \
                 second_level_shape[1:3]
        
        #var_length_extras = [group_differences1, group_differences2]
#         var_length_bits = [first_level_max_group_size_bits,  
#                            second_level_max_group_size_bits]
        var_length_extras = []
        var_length_bits = []
        
        var_length_extras += outlier_extras2
        var_length_bits += [ outlier_index_bytes * 8, outlier_sample_bytes * 8 ]
            
        if use_importance_sampling:

            var_length_extras += outlier_extras1
            var_length_bits += [ outlier_index_bytes * 8, outlier_sample_bytes * 8 ]
            
        #np.load(second_level_counts)
        second_level_coder = ArithmeticCoder(np.ones(2**7 + 1), precision=32)
        #np.load(first_level_counts)
        first_level_coder = ArithmeticCoder(np.ones(2**7 + 1), precision=32)
        
        group_differences1 = np.concatenate((group_differences1, [0]))
        gi1_code = first_level_coder.encode(group_differences1)
        print(len(gi1_code))
        print(20 * len(group_differences1))
        
        group_differences2 = np.concatenate((group_differences2, [0]))
        gi2_code = second_level_coder.encode(group_differences2)

        write_bin_code(bitcode, 
                       comp_file_path, 
                       extras=extras,
                       extra_var_bits=[gi1_code, gi2_code],
                       var_length_extras=var_length_extras,
                       var_length_bits=var_length_bits)
        
        # -------------------------------------------------------------------------------------
        # Step 4: Some logging information
        # -------------------------------------------------------------------------------------
        
        total_kls = [tf.reduce_sum(x) for x in (self.first_level_kl, self.second_level_kl)]
        total_kls = session.run(total_kls)
        total_kl = sum(total_kls)

        theoretical_byte_size = (total_kl + 2 * np.log(total_kl + 1)) / np.log(2) / 8
        extra_byte_size = len(gi1_code) + len(gi2_code) + 9 * 2 // 8
        actual_byte_size = os.path.getsize(comp_file_path)

        actual_no_extra = actual_byte_size - extra_byte_size
        
        first_level_theoretical = (total_kls[0] + 2 * np.log(total_kls[0] + 1)) / np.log(2) / 8
        first_level_actual_no_extra = len(code1) / 8
        first_level_extra = len(gi1_code) // 8

        sample1_reshaped = tf.reshape(sample1, first_level_shape)
        first_level_avg_log_lik = tf.reduce_mean(self.posterior_1.log_prob(sample1_reshaped))
        first_level_sample_avg = tf.reduce_mean(self.posterior_1.log_prob(self.posterior_1.sample()))
        
        first_level_avg_log_lik, first_level_sample_avg = session.run([first_level_avg_log_lik, first_level_sample_avg])
        
        
        second_level_theoretical = (total_kls[1] + 2 * np.log(total_kls[1] + 1)) / np.log(2) / 8
        second_level_actual_no_extra = len(code2) / 8
        second_level_extra = len(gi2_code) // 8 + 1
        
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
                            second_level_counts="/homes/gf332/compression-without-quantization/group_dists_2.npy",
                            first_level_counts="/homes/gf332/compression-without-quantization/group_dists_1.npy",
                            verbose=False):
        
        # -------------------------------------------------------------------------------------
        # Step 1: Read the compressed file
        # -------------------------------------------------------------------------------------
        
        # the extras are: seed, n_steps, n_bits_per_step and W x H of the two latent levels
        # var length extras are the two lists of group indices
        # +2: outlier extras of the second level importance sampler
        num_var_length_extras = 2
        
        if use_importance_sampling:
            num_var_length_extras += 2
        
        code, extras, extra_var_bits, var_length_extras = read_bin_code(comp_file_path, 
                                                                        num_extras=9, 
                                                                        num_extra_var_bits=2,
                                                                        num_var_length_extras=num_var_length_extras)
        
        second_level_coder = ArithmeticCoder(np.load(second_level_counts), precision=32)
        first_level_coder = ArithmeticCoder(np.load(first_level_counts), precision=32)
               
        print(extras)
        
        seed = extras[0]
        
        n_steps = extras[1]
        n_bits_per_step = extras[2]
        first_level_n_bits_per_group = extras[3]
        second_level_n_bits_per_group = extras[4]
        
        # Get shape information back
        first_level_shape = [1] + extras[5:7] + [self.first_level_latent_channels]
        second_level_shape = [1] + extras[7:] + [self.second_level_latent_channels]
        
        # Total number of latents on levels
        num_first_level = np.prod(first_level_shape)
        num_second_level = np.prod(second_level_shape)
        
        # Remember to chop off the terminating EOF 0
        group_differences2 = second_level_coder.decode_fast(extra_var_bits[1])[:-1]
        group_differences1 = first_level_coder.decode_fast(extra_var_bits[0])[:-1]
        print(group_differences1[-30:])
        
        first_code_length = first_level_n_bits_per_group * len(group_differences1)
        second_code_length = second_level_n_bits_per_group * len(group_differences2)
        
        print(first_code_length)
        print(second_code_length) 
        
        code1 = code[:first_code_length]
        code2 = code[first_code_length:first_code_length + second_code_length]
        
        print(len(code1))
        print(len(code2))
        # -------------------------------------------------------------------------------------
        # Step 2: Decode the samples
        # -------------------------------------------------------------------------------------
        
        # Decode second level
        proposal = tfd.Normal(loc=tf.zeros(second_level_shape),
                              scale=tf.ones(second_level_shape))
        
        
        # Get group indices back
        #group_differences2 = var_length_extras[1]

        group_indices2 = [0]
        
        for i in range(0, len(group_differences2)):
            # The +1 at the end shifts the group sizes back from the range 0 - 15 to the range 1 - 16.
            group_indices2.append(group_indices2[i] + group_differences2[i])
           
        
        print("Decoding second level")
        decoded_second_level = decode_grouped_importance_sample(sess=session,
                                                                bitcode=code2, 
                                                                group_start_indices=group_indices2[:-1],
                                                                proposal=proposal, 
                                                                n_bits_per_group=second_level_n_bits_per_group,
                                                                seed=seed,
                                                                outlier_indices=var_length_extras[0],
                                                                outlier_samples=var_length_extras[1])

        np.save("/homes/gf332/compression-without-quantization/s2_decod.npy", decoded_second_level)
        decoded_second_level = tf.reshape(decoded_second_level, second_level_shape)
        
        # Now we can calculate the the first level priors
        ops = [self.synthesis_transform_2(decoded_second_level),
               self.synthesis_transform_2.loc, 
               self.synthesis_transform_2.scale]

        _, p1_loc, p1_scale = session.run(ops)
        
        self.prior_1 = tfd.Normal(loc=p1_loc, scale=p1_scale)
        
        group_indices1 = [0]
        
        for i in range(0, len(group_differences1)):
            group_indices1.append(group_indices1[i] + group_differences1[i]) 
        
        # Decode first level
        print("Decoding first level")
        if use_importance_sampling:
            
            print(group_indices1[:30])
            print(code1[:70])
            print(group_indices1[-30:])
            print(code1[-70:])
            print(first_level_n_bits_per_group)
            decoded_first_level = decode_grouped_importance_sample(sess=session,
                                                                bitcode=code1, 
                                                                group_start_indices=group_indices1[:-1],
                                                                proposal=self.prior_1, 
                                                                n_bits_per_group=first_level_n_bits_per_group,
                                                                seed=seed,
                                                                outlier_indices=var_length_extras[2],
                                                                outlier_samples=var_length_extras[3])
            

        else:
            decoded_first_level = decode_grouped_greedy_sample(sess=session,
                                                               bitcode=code1, 
                                                                group_start_indices=group_indices1,
                                                                proposal=self.prior_1, 
                                                                n_bits_per_step=n_bits_per_step, 
                                                                n_steps=n_steps, 
                                                                seed=seed,
                                                                rho=rho,
                                                                adaptive=True)
            
        np.save("/homes/gf332/compression-without-quantization/s1_decod.npy", decoded_first_level)
        decoded_first_level = tf.reshape(decoded_first_level, first_level_shape)
        

        # -------------------------------------------------------------------------------------
        # Step 4: Reconstruct the image with the VAE
        # -------------------------------------------------------------------------------------
        
        reconstruction = self.synthesis_transform_1(decoded_first_level)
        
        return session.run(tf.squeeze(reconstruction))
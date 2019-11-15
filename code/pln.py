import os

import pickle

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


        # TODO understand this.
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
        # self.posterior_1 = tfd.Normal(loc=combined_loc,
        #                               scale=combined_scale)
        #
        # self.prior_1 = self.synthesis_transform_2.prior
        #
        # self.posterior_2 = self.analysis_transform_2.posterior
        # self.prior_2 = self.analysis_transform_2.prior
        self.posterior_1 = tfd.Normal(loc=self.analysis_transform_1.loc,
                                      scale=self.analysis_transform_1.scale)

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
                          use_permutation=True,
                          
                          # Importance sampling parameters
                          second_level_n_bits_per_group=20,
                          second_level_max_group_size_bits=4,
                          second_level_dim_kl_bit_limit=12,
                          first_level_n_bits_per_group=20,
                          first_level_max_group_size_bits=3,
                          first_level_dim_kl_bit_limit=12,
                          outlier_index_bytes=3,
                          outlier_sample_bytes=2,
                          
                          second_level_group_dist_counts="",
                          first_level_group_dist_counts="",
                          
                          second_level_sample_index_counts="",
                          first_level_sample_index_counts="",
                          
                          second_level_sample_ac="/homes/gf332/compression-without-quantization/samp_ind_2_ac.pkl",
                          first_level_sample_ac="/homes/gf332/compression-without-quantization/samp_ind_1_ac.pkl",
                          
                          use_index_ac = False,
                          
                          return_first_level_group_sizes=False,
                          return_first_level_indices=False,
                          
                          return_second_level_group_sizes=False,
                          return_second_level_indices=False,
                          
                          verbose=False):
        
        flatten = lambda x: tf.reshape(x, [-1])
        
        if use_index_ac:
            print("loading arithmetic coders for index coding")
            ind_ac_1 = pickle.load(open(first_level_sample_ac, "rb"))
            ind_ac_2 = pickle.load(open(second_level_sample_ac, "rb"))
        
        # -------------------------------------------------------------------------------------
        # Step 1: Set the latent distributions for the image
        # -------------------------------------------------------------------------------------
        
        if verbose: print("Calculating latent distributions for image...")

        q1_loc_op = flatten(self.posterior_1.loc)
        q1_scale_op = flatten(self.posterior_1.scale)
        
        q2_loc_op = flatten(self.posterior_2.loc)
        q2_scale_op = flatten(self.posterior_2.scale)
        
        p2_loc_op = flatten(self.prior_2.loc)
        p2_scale_op = flatten(self.prior_2.scale)
        
        first_level_shape_op = tf.shape(self.prior_1.loc)
        second_level_shape_op = tf.shape(self.prior_2.loc)
            
        ops = [self.call(image),
               q1_loc_op,
               q1_scale_op,
               q2_loc_op,
               q2_scale_op,
               p2_loc_op,
               p2_scale_op,
               first_level_shape_op,
               second_level_shape_op]
        
        im, q1_loc, q1_scale, q2_loc, q2_scale, p2_loc, p2_scale, first_level_shape, second_level_shape = session.run(ops)
        
#         print(q1_loc.shape)
#         print(im.shape)
        
        # We will randomly permute the dimensions to destroy any structure present in the vector
        np.random.seed(seed)
        
        num_dim_1 = np.prod(q1_loc.shape)
        num_dim_2 = np.prod(q2_loc.shape)
        
        perm_1 = np.random.permutation(num_dim_1).astype("int32") \
                if use_permutation else np.arange(num_dim_1, dtype=np.int32)
        perm_2 = np.random.permutation(num_dim_2).astype("int32") \
                if use_permutation else np.arange(num_dim_2, dtype=np.int32)
        
        permuter_1 = tfp.bijectors.Permute(permutation=perm_1)     
        permuter_2 = tfp.bijectors.Permute(permutation=perm_2)
        
        q1_loc_op = permuter_1.forward(q1_loc)
        q1_scale_op = permuter_1.forward(q1_scale)
        
        q2_loc_op = permuter_2.forward(q2_loc)
        q2_scale_op = permuter_2.forward(q2_scale)
        
        p2_loc_op = permuter_2.forward(p2_loc)
        p2_scale_op = permuter_2.forward(p2_scale)
        
        ops = [q1_loc_op,
               q1_scale_op,
               q2_loc_op,
               q2_scale_op,
               p2_loc_op,
               p2_scale_op]

        q1_loc_, q1_scale_, q2_loc_, q2_scale_, p2_loc_, p2_scale_ = session.run(ops)


#         q1 = tfd.Normal(loc=permuter_1.forward(q1_loc), scale=permuter_1.forward(q1_scale))
        q2 = tfd.Normal(loc=q2_loc_, scale=q2_scale_)
        p2 = tfd.Normal(loc=p2_loc_, scale=p2_scale_)
            
        q1 = tfd.Normal(loc=q1_loc_, scale=q1_scale_)
#         q2 = tfd.Normal(loc=q2_loc, scale=q2_scale)
#         p2 = tfd.Normal(loc=p2_loc, scale=p2_scale)
        
        image_shape = list(im.shape)
        print(first_level_shape)
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
                                            return_group_indices_only=return_second_level_group_sizes,
                                            return_indices_only=return_second_level_indices,
                                            return_indices=use_index_ac)
        
        
        if return_second_level_group_sizes:
            group_start_indices, group_kls = res
            
            return group_start_indices
        
        elif return_second_level_indices:
            
            return res

        sample2, code2, group_indices2, outlier_extras2 = res
        
        if use_index_ac:
            
            code2 = ''.join(ind_ac_2.encode(code2))
            
            #print(code2)
            
            #return

#         print(outlier_extras2)
        
        outlier_extras2 = list(map(lambda x: x.reshape([-1]), outlier_extras2))
        
        # The -1 at the end shifts the range of the group sizes to the 0 - 15 range from the 1 - 16 range
        # such that it can be coded as usual
        
        group_differences2 = group_indices2[1:] - group_indices2[:-1]
        
        group_indices2_ = [0]
        
        for i in range(0, len(group_differences2)):
            # The +1 at the end shifts the group sizes back from the range 0 - 15 to the range 1 - 16.
            group_indices2_.append(group_indices2_[i] + (group_differences2[i] + 1))
         
        # Reverse the permutation so that we can pass the sample through the second level
        sample2 = permuter_2.inverse(sample2)
        
        # We need to adjust the priors to the second stage sample
        latents = tf.reshape(sample2, second_level_shape)
        
        p1_loc_op = flatten(self.synthesis_transform_2.loc)
        p1_scale_op = flatten(self.synthesis_transform_2.scale)
        
        ops = [self.synthesis_transform_2(latents), 
               permuter_1.forward(p1_loc_op), 
               permuter_1.forward(p1_scale_op)]
        
        _, p1_loc, p1_scale = session.run(ops)
        
        p1 = tfd.Normal(loc=p1_loc, 
                        scale=p1_scale)
        
#         p1 = tfd.Normal(loc=p1_loc, 
#                         scale=p1_scale)
        
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
                                                return_group_indices_only=return_first_level_group_sizes,
                                                return_indices_only=return_first_level_indices,
                                                return_indices=use_index_ac)
            
            if return_first_level_group_sizes:
                group_start_indices, group_kls = res

                return group_start_indices
            
            elif return_first_level_indices:
                
                return res
            
            sample1, code1, group_indices1, outlier_extras1 = res
            
            if use_index_ac:
                code1 = ''.join(ind_ac_1.encode(code1))
            
            unique, counts = np.unique(group_indices1, return_counts=True)
            
            group_index_counts[unique] += counts
            
            np.save("gic.npy", group_index_counts)
            
#             print(outlier_extras1)
            
#             print(group_indices1[:30])
#             print(group_indices1[-30:])
#             print(code1[:70])
#             print(code1[-70:])
#             print(first_level_n_bits_per_group)
            
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

        if use_index_ac:
            bitcode = code1 + code2
        else:
            bitcode = code1.decode("utf-8") if use_importance_sampling else code1
            bitcode += code2.decode("utf-8")
        
#         print(len(code1))
#         print(len(code2))
        # -------------------------------------------------------------------------------------
        # Step 3: Write the compressed file
        # -------------------------------------------------------------------------------------
        
        extras = [seed, 
                  n_steps, 
                  n_bits_per_step, 
                  first_level_n_bits_per_group, 
                  second_level_n_bits_per_group,
                  len(code1),
                  len(code2)
                 ] + \
                 list(first_level_shape[1:3]) + \
                 list(second_level_shape[1:3])
        
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
            
        #second_level_sample_index_counts
        second_level_group_size_coder = ArithmeticCoder(np.load(second_level_group_dist_counts), precision=32)
        first_level_group_size_coder = ArithmeticCoder(np.load(first_level_group_dist_counts), precision=32)
        
#         second_level_index_coder = ArithmeticCoder(np.load(second_level_sample_index_counts), precision=32)
#         first_level_index_coder = ArithmeticCoder(np.load(first_level_sample_index_counts), precision=32)
        
        group_differences1 = np.concatenate((group_differences1, [0]))
        gi1_code = first_level_group_size_coder.encode(group_differences1)
#         print(len(gi1_code))
#         print(20 * len(group_differences1))
        
        group_differences2 = np.concatenate((group_differences2, [0]))
        gi2_code = second_level_group_size_coder.encode(group_differences2)

#         print(extras)
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

        sample1_reshaped = tf.reshape(permuter_1.inverse(sample1), first_level_shape)
        first_level_avg_log_lik = tf.reduce_mean(self.posterior_1.log_prob(sample1_reshaped))
        first_level_sample_avg = tf.reduce_mean(self.posterior_1.log_prob(self.posterior_1.sample()))
        
        first_level_avg_log_lik, first_level_sample_avg = session.run([first_level_avg_log_lik, first_level_sample_avg])
        
        
        second_level_theoretical = (total_kls[1] + 2 * np.log(total_kls[1] + 1)) / np.log(2) / 8
        second_level_actual_no_extra = len(code2) / 8
        second_level_extra = len(gi2_code) // 8 + 1
        
        second_bpp = (second_level_actual_no_extra + second_level_extra) * 8 / (image_shape[1] * image_shape[2]) 

        sample2_reshaped = tf.reshape(permuter_2.inverse(sample2), second_level_shape)
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
                            use_permutation=True,
                            second_level_group_dist_counts="",
                            first_level_group_dist_counts="",
                            second_level_sample_ac="/homes/gf332/compression-without-quantization/samp_ind_2_ac.pkl",
                            first_level_sample_ac="/homes/gf332/compression-without-quantization/samp_ind_1_ac.pkl",
                            use_index_ac=False,
                            verbose=False):
        
        if use_index_ac:
            print("loading arithmetic coders for index coding")
            ind_ac_1 = pickle.load(open(first_level_sample_ac, "rb"))
            ind_ac_2 = pickle.load(open(second_level_sample_ac, "rb"))
        
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
                                                                        num_extras=11, 
                                                                        num_extra_var_bits=2,
                                                                        num_var_length_extras=num_var_length_extras)
        
        
        second_level_coder = ArithmeticCoder(np.load(second_level_group_dist_counts), precision=32)
        first_level_coder = ArithmeticCoder(np.load(first_level_group_dist_counts), precision=32)
               
        print("Extras: {}".format(extras))
        
        seed = extras[0]
        
        n_steps = extras[1]
        n_bits_per_step = extras[2]
        first_level_n_bits_per_group = extras[3]
        second_level_n_bits_per_group = extras[4]
        first_code_length = extras[5]
        second_code_length = extras[6]
        
        # Get shape information back
        first_level_shape = [1] + extras[7:9] + [self.first_level_latent_channels]
        second_level_shape = [1] + extras[9:] + [self.second_level_latent_channels]
        
        # Total number of latents on levels
        num_first_level = np.prod(first_level_shape)
        num_second_level = np.prod(second_level_shape)
        
        # Set up permutation to back-permute samples
        np.random.seed(seed)
        
        perm_1 = np.random.permutation(num_first_level).astype("int32") \
                if use_permutation else np.arange(num_first_level, dtype=np.int32)
        perm_2 = np.random.permutation(num_second_level).astype("int32") \
                if use_permutation else np.arange(num_second_level, dtype=np.int32)
        
        permuter_1 = tfp.bijectors.Permute(permutation=perm_1)     
        permuter_2 = tfp.bijectors.Permute(permutation=perm_2)
        
        # Remember to chop off the terminating EOF 0
        group_differences2 = second_level_coder.decode_fast(extra_var_bits[1])[:-1]
        group_differences1 = first_level_coder.decode_fast(extra_var_bits[0])[:-1]
        print(group_differences1[-30:])
        
#         print(first_code_length)
#         print(second_code_length) 
        
        code1 = code[:first_code_length]
        code2 = code[first_code_length:first_code_length + second_code_length]
        
#         print(len(code1))
#         print(len(code2))
        
        if use_index_ac:
            code1 = ind_ac_1.decode_fast(code1)
            code2 = ind_ac_2.decode_fast(code2)
        # -------------------------------------------------------------------------------------
        # Step 2: Decode the samples
        # -------------------------------------------------------------------------------------
        
        # Decode second level
        proposal = tfd.Normal(loc=tf.zeros(num_second_level), #tf.zeros(second_level_shape),
                              scale=tf.ones(num_second_level)) #tf.ones(second_level_shape))
        
        
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
                                                                outlier_samples=var_length_extras[1],
                                                                use_indices=use_index_ac)

        np.save("/homes/gf332/compression-without-quantization/s2_decod.npy", decoded_second_level)
        
        # Reverse the permutation
        decoded_second_level = permuter_2.inverse(decoded_second_level)
        decoded_second_level = tf.reshape(decoded_second_level, second_level_shape)
        
        # Now we can calculate the the first level priors
        ops = [self.synthesis_transform_2(decoded_second_level),
               permuter_1.forward(tf.reshape(self.synthesis_transform_2.loc, [-1])), 
               permuter_1.forward(tf.reshape(self.synthesis_transform_2.scale, [-1]))]

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
                                                                outlier_samples=var_length_extras[3],
                                                                use_indices=use_index_ac)
            

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
        # Reverse the permutation
        decoded_first_level = permuter_1.inverse(decoded_first_level)
        decoded_first_level = tf.reshape(decoded_first_level, first_level_shape)
        

        # -------------------------------------------------------------------------------------
        # Step 4: Reconstruct the image with the VAE
        # -------------------------------------------------------------------------------------
        
        reconstruction = self.synthesis_transform_1(decoded_first_level)
        
        return session.run(tf.squeeze(reconstruction))

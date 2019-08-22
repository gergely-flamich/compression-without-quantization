import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras
tfl = tf.keras.layers

from losses import SsimMultiscalePseudo, MixPseudo

from transforms import AnalysisTransform_1, SynthesisTransform_1

from coded_greedy_sampler import code_grouped_greedy_sample, decode_grouped_greedy_sample
from coded_importance_sampler import code_grouped_importance_sample, decode_grouped_importance_sample

from binary_io import write_bin_code, read_bin_code

# =====================================================================
# =====================================================================
# VAE definition
# =====================================================================
# =====================================================================

class VariationalAutoEncoder(tfk.Model):
    
    def __init__(self,
                 num_filters,
                 num_latent_channels,
                 padding="same_zeros",
                 likelihood="gaussian",
                 learn_gamma=False,
                 *args,
                 **kwargs):
        
        self.num_filters = num_filters
        self.num_latent_channels = num_latent_channels
        self.padding = padding
        
        self.learn_gamma = learn_gamma
        
        if likelihood == "gaussian":
            self.likelihood_dist_family = tfd.Normal
            
        elif likelihood == "laplace":
            self.likelihood_dist_family = tfd.Laplace
            
        else:
            raise Exception("Unknown likelihood: {}".format(likelihood))
        
        super(VariationalAutoEncoder, self).__init__(*args, **kwargs)
        
        
    @property
    def kl_divergence(self):
        return tfd.kl_divergence(self.posterior, self.prior)    
        
    def bpp(self, num_pixels):
        return tf.reduce_sum(self.kl_divergence) / (np.log(2) * num_pixels)
    
    def save_latent_statistics(self, sess, dir_path):
        
        q_loc = self.posterior.loc.eval(session = sess)
        q_scale = self.posterior.scale.eval(session = sess)
        
        p_loc = self.prior.loc.eval(session = sess)
        p_scale = self.prior.scale.eval(session = sess)
        
        np.save(dir_path + "/vae_q_loc.npy", q_loc)
        np.save(dir_path + "/vae_q_scale.npy", q_scale)
        np.save(dir_path + "/vae_p_loc.npy", p_loc)
        np.save(dir_path + "/vae_p_scale.npy", p_scale)
        
    def build(self, input_shape):
        
        self.analysis_transform = AnalysisTransform_1(num_filters=self.num_filters,
                                                      num_latent_channels=self.num_latent_channels,
                                                      padding=self.padding)
        
        self.synthesis_transform = SynthesisTransform_1(num_filters=self.num_filters,
                                                        padding=self.padding)
        
        if self.learn_gamma:
            self.log_gamma = self.add_variable("log_gamma", 
                                               dtype=tf.float32, 
                                               initializer=tf.compat.v1.constant_initializer(0.),
                                               trainable=True)
            
        else:
            self.log_gamma = tf.constant(0., dtype=tf.float32)
        
        super(VariationalAutoEncoder, self).build(input_shape)

        
    def call(self, inputs):
        
        # Perform analysis pass
        z = self.analysis_transform(inputs)
        
        # Perform reconstruction
        reconstruction = self.synthesis_transform(z)
        
        # Set latent distributions
        self.prior = self.analysis_transform.prior
        self.posterior = self.analysis_transform.posterior
        
        # Likelihood
        self.likelihood_dist = self.likelihood_dist_family(loc=reconstruction,
                                                           scale=tf.math.exp(self.log_gamma))
        
        self.log_likelihood = self.likelihood_dist.log_prob(inputs)
        
        return reconstruction
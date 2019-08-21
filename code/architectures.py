import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras
tfl = tf.keras.layers

from losses import SsimMultiscalePseudo, MixPseudo

# =====================================================================
# =====================================================================
# First Level
# =====================================================================
# =====================================================================

class AnalysisTransform_1(tfl.Layer):
    """
    Level 1 in the PLN
    """
    
    def __init__(self, 
                 num_filters, 
                 num_latent_channels,
                 padding="same_zeros",
                 *args,
                 **kwargs):
        
        self.num_filters = num_filters
        self.num_latent_channels = num_latent_channels
        self.padding = padding
        
        super(AnalysisTransform_1, self).__init__(*args, **kwargs)
        
        
    def build(self, input_shape):
        
        self.layers = [
            tfc.SignalConv2D(filters=self.num_filters, 
                             kernel_support=(5, 5), 
                             name="layer_0", 
                             corr=True, 
                             strides_down=2,
                             padding=self.padding, 
                             use_bias=True,
                             activation=tfc.GDN(name="gdn_0")),
            tfc.SignalConv2D(filters=self.num_filters, 
                             kernel_support=(5, 5), 
                             name="layer_1", 
                             corr=True, 
                             strides_down=2,
                             padding=self.padding, 
                             use_bias=True,
                             activation=tfc.GDN(name="gdn_1")),
            tfc.SignalConv2D(filters=self.num_filters, 
                             kernel_support=(5, 5), 
                             name="layer_2", 
                             corr=True, 
                             strides_down=2, 
                             padding=self.padding, 
                             use_bias=True,
                             activation=tfc.GDN(name="gdn_2"))]
        
        
        # Note the linear activation
        self.loc_head = tfc.SignalConv2D(filters=self.num_latent_channels, 
                                         kernel_support=(5, 5), 
                                         name="layer_loc", 
                                         corr=True, 
                                         strides_down=2,
                                         padding=self.padding, 
                                         use_bias=False,
                                         activation=None)
        
        # Note the exp activation
        self.scale_head = tfc.SignalConv2D(filters=self.num_latent_channels, 
                                           kernel_support=(5, 5), 
                                           name="layer_scale", 
                                           corr=True, 
                                           strides_down=2,
                                           padding=self.padding, 
                                           use_bias=False,
                                           activation=tf.math.exp)
        
        super(AnalysisTransform_1, self).build(input_shape)
          
            
    def call(self, activations):
        
        # Omit the loc and scale heads
        for layer in self.layers:
            activations = layer(activations)
            
        self.loc = self.loc_head(activations)
        self.scale = self.scale_head(activations)
        
        self.posterior = tfd.Normal(loc=self.loc,
                                    scale=self.scale)
        
        self.prior = tfd.Normal(loc=tf.zeros_like(self.loc),
                                scale=tf.ones_like(self.scale))
        
        return self.posterior.sample()
    
    
    
class SynthesisTransform_1(tfl.Layer):
    """
    Level 1 in the PLN
    """
    
    def __init__(self, 
                 num_filters,
                 padding="same_zeros",
                 *args,
                 **kwargs):
        
        self.num_filters = num_filters
        self.padding = padding
        
        super(SynthesisTransform_1, self).__init__(*args, **kwargs)
        
    def build(self, input_shape):
        
        self.layers = [
            tfc.SignalConv2D(filters=self.num_filters, 
                             kernel_support=(5, 5), 
                             name="layer_0", 
                             corr=False, 
                             strides_up=2,
                             padding=self.padding, 
                             use_bias=True,
                             activation=tfc.GDN(name="igdn_0", inverse=True)),
            
            tfc.SignalConv2D(filters=self.num_filters, 
                             kernel_support=(5, 5), 
                             name="layer_1", 
                             corr=False, 
                             strides_up=2,
                             padding=self.padding, 
                             use_bias=True,
                             activation=tfc.GDN(name="igdn_1", inverse=True)),
            
            tfc.SignalConv2D(filters=self.num_filters, 
                             kernel_support=(5, 5), 
                             name="layer_2", 
                             corr=False, 
                             strides_up=2,
                             padding=self.padding, 
                             use_bias=True,
                             activation=tfc.GDN(name="igdn_2", inverse=True)),
            
            # The output always has 3 channels
            tfc.SignalConv2D(filters=3, 
                             kernel_support=(5, 5), 
                             name="layer_3", 
                             corr=False, 
                             strides_up=2,
                             padding=self.padding, 
                             use_bias=True,
                             activation=None),
           # tf.nn.sigmoid
        ]
        
        super(SynthesisTransform_1, self).build(input_shape)

    def call(self, activations):
        
        for layer in self.layers:
            activations = layer(activations)
            
        return activations
    
    
# =====================================================================
# =====================================================================
# Second Level
# =====================================================================
# =====================================================================

class AnalysisTransform_2(tfl.Layer):
    """Second Level"""

    def __init__(self, 
                 num_filters, 
                 num_latent_channels,
                 padding="same_zeros",
                 *args, 
                 **kwargs):
        
        self.num_filters = num_filters
        self.padding = padding
        
        super(AnalysisTransform_2, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        
        self.layers = [
            tfc.SignalConv2D(filters=self.num_filters, 
                             kernel_support=(3, 3),
                             name="layer_0",
                             corr=True,
                             strides_down=1,
                             padding=self.padding,
                             use_bias=True,
                             activation=tf.nn.leaky_relu),
            
            tfc.SignalConv2D(filters=self.num_filters, 
                             kernel_support=(5, 5),
                             name="layer_1",
                             corr=True,
                             strides_down=2,
                             padding=self.padding,
                             use_bias=True,
                             activation=tf.nn.leaky_relu),
        ]
        
        # Note the linear activation
        self.loc_head = tfc.SignalConv2D(filters=self.num_filters, 
                                         kernel_support=(5, 5),
                                         name="layer_loc",
                                         corr=True,
                                         strides_down=2,
                                         padding=self.padding,
                                         use_bias=False,
                                         activation=None)
        
        # Note the sigmoid activation
        self.scale_head = tfc.SignalConv2D(filters=self.num_filters, 
                                           kernel_support=(5, 5),
                                           name="layer_scale",
                                           corr=True,
                                           strides_down=2,
                                           padding=self.padding,
                                           use_bias=False,
                                           activation=tf.nn.sigmoid)
        
        super(AnalysisTransform_2, self).build(input_shape)

        
    def call(self, inputs):
        
        activations = inputs
        
        # Omit the loc and scale heads
        for layer in self.layers:
            activations = layer(activations)
            
        self.loc = self.loc_head(activations)
        self.scale = self.scale_head(activations)
        
        self.posterior = tfd.Normal(loc=self.loc,
                                    scale=self.scale)
        
        self.prior = tfd.Normal(loc=tf.zeros_like(self.loc),
                                scale=tf.ones_like(self.scale))
        
        return self.posterior.sample()


class SynthesisTransform_2(tfl.Layer):
    """
    Second Level
    """

    def __init__(self, 
                 num_filters, 
                 num_output_channels,
                 padding="same_zeros",
                 *args, 
                 **kwargs):
        
        self.num_filters = num_filters
        self.num_output_channels = num_output_channels
        self.padding = padding
        
        super(SynthesisTransform_2, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.layers = [
            tfc.SignalConv2D(filters=self.num_filters, 
                             kernel_support=(5, 5), 
                             name="layer_0", 
                             corr=False, 
                             strides_up=2, 
                             padding=self.padding, 
                             use_bias=True, 
                             kernel_parameterizer=None,
                             activation=tf.nn.leaky_relu),
            tfc.SignalConv2D(filters=self.num_filters, 
                             kernel_support=(5, 5), 
                             name="layer_1", 
                             corr=False, 
                             strides_up=2, 
                             padding=self.padding, 
                             use_bias=True, 
                             kernel_parameterizer=None,
                             activation=tf.nn.leaky_relu),
        ]
        
        # Note the linear activation
        self.loc_head = tfc.SignalConv2D(filters=self.num_output_channels, 
                                         kernel_support=(3, 3), 
                                         name="layer_loc", 
                                         corr=False, 
                                         strides_up=1, 
                                         padding=self.padding, 
                                         use_bias=True, 
                                         kernel_parameterizer=None,
                                         activation=None)
        
        # Note the softplus activation
        self.scale_head = tfc.SignalConv2D(filters=self.num_output_channels, 
                                           kernel_support=(3, 3), 
                                           name="layer_scale", 
                                           corr=False, 
                                           strides_up=1, 
                                           padding=self.padding, 
                                           use_bias=True, 
                                           kernel_parameterizer=None,
                                           activation=tf.nn.softplus)
        
        super(SynthesisTransform_2, self).build(input_shape)

        
    def call(self, inputs):
        
        activations = inputs
        
        # Omit the loc and scale heads
        for layer in self.layers:
            activations = layer(activations)
            
        self.loc = self.loc_head(activations)
        self.scale = self.scale_head(activations)
        
        self.prior = tfd.Normal(loc=self.loc, 
                                scale=self.scale)
            
        return self.prior.sample()
    
    
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
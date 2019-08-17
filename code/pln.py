import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras
tfl = tf.keras.layers

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
                 *args, 
                 padding="same_zeros",
                 **kwargs):
        
        self.num_filters = num_filters
        self.num_latent_channels = num_latent_channels
        self.padding = padding
        
        super(AnalysisTransform_1, self).__init__(*args, **kwargs)
        
        
    def build(self, input_shape):
        
        self._layers = [
            tfc.SignalConv2D(self.num_filters, 
                             (5, 5), 
                             name="layer_0", 
                             corr=True, 
                             strides_down=2,
                             padding=self.padding, 
                             use_bias=True,
                             activation=tfc.GDN(name="gdn_0")),
            tfc.SignalConv2D(self.num_filters, 
                             (5, 5), 
                             name="layer_1", 
                             corr=True, 
                             strides_down=2,
                             padding=self.padding, 
                             use_bias=True,
                             activation=tfc.GDN(name="gdn_1")),
            tfc.SignalConv2D(self.num_filters, 
                             (5, 5), 
                             name="layer_2", 
                             corr=True, 
                             strides_down=2, 
                             padding=self.padding, 
                             use_bias=True,
                             activation=tfc.GDN(name="gdn_2"))]
        
        
        # Note the linear activation
        self.loc_head = tfc.SignalConv2D(self.num_latent_channels, 
                                          (5, 5), 
                                          name="layer_loc", 
                                          corr=True, 
                                          strides_down=2,
                                          padding=self.padding, 
                                          use_bias=True,
                                          activation=None)
        
        # Note the exponential activation
        self.scale_head = tfc.SignalConv2D(self.num_latent_channels, 
                                            (5, 5), 
                                            name="layer_scale", 
                                            corr=True, 
                                            strides_down=2,
                                            padding=self.padding, 
                                            use_bias=True,
                                            activation=tf.math.exp)
        
        super(AnalysisTransform_1, self).build(input_shape)
          
            
    def call(self, inputs):
        
        activations = inputs
        
        # Omit the loc and scale heads
        for layer in self._layers[:-2]:
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
                 *args, 
                 padding="same_zeros",
                 **kwargs):
        
        self.num_filters = num_filters
        self.padding = padding
        
        super(SynthesisTransform_1, self).__init__(*args, **kwargs)
        
    def build(self, input_shape):
        
        self._layers = [
            tfc.SignalConv2D(self.num_filters, 
                             (5, 5), 
                             name="layer_0", 
                             corr=False, 
                             strides_up=2,
                             padding=self.padding, 
                             use_bias=True,
                             activation=tfc.GDN(name="igdn_0", inverse=True)),
            
            tfc.SignalConv2D(self.num_filters, 
                             (5, 5), 
                             name="layer_1", 
                             corr=False, 
                             strides_up=2,
                             padding=self.padding, 
                             use_bias=True,
                             activation=tfc.GDN(name="igdn_1", inverse=True)),
            
            tfc.SignalConv2D(self.num_filters, 
                             (5, 5), 
                             name="layer_2", 
                             corr=False, 
                             strides_up=2,
                             padding=self.padding, 
                             use_bias=True,
                             activation=tfc.GDN(name="igdn_2", inverse=True)),
            
            # The output always has 3 channels
            tfc.SignalConv2D(3, 
                             (5, 5), 
                             name="layer_3", 
                             corr=False, 
                             strides_up=2,
                             padding=self.padding, 
                             use_bias=True,
                             activation=None)]
        
        super(SynthesisTransform_1, self).build(input_shape)

    def call(self, inputs):
        
        activations = inputs
        
        for layer in self._layers:
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
                 *args, 
                 padding="same_zeros",
                 **kwargs):
        
        self.num_filters = num_filters
        self.padding = padding
        
        super(AnalysisTransform_2, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        
        self._layers = [
            tfc.SignalConv2D(self.num_filters, 
                             (3, 3),
                             name="layer_0",
                             corr=True,
                             strides_down=1,
                             padding=self.padding,
                             use_bias=True,
                             activation=tf.nn.relu),
            
            tfc.SignalConv2D(self.num_filters, 
                             (5, 5),
                             name="layer_1",
                             corr=True,
                             strides_down=2,
                             padding=self.padding,
                             use_bias=True,
                             activation=tf.nn.relu),
        ]
        
        # Note the linear activation
        self.loc_head = tfc.SignalConv2D(self.num_filters, 
                                         (5, 5),
                                         name="layer_loc",
                                         corr=True,
                                         strides_down=2,
                                         padding=self.padding,
                                         use_bias=False,
                                         activation=None)
        
        # Note the sigmoid activation
        self.scale_head = tfc.SignalConv2D(self.num_filters, 
                                           (5, 5),
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
        for layer in self._layers[:-2]:
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
                 *args, 
                 padding="same_zeros",
                 **kwargs):
        
        self.num_filters = num_filters
        self.num_output_channels = num_output_channels
        self.padding = padding
        
        super(SynthesisTransform_2, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(self.num_filters, 
                             (5, 5), 
                             name="layer_0", 
                             corr=False, 
                             strides_up=2, 
                             padding=self.padding, 
                             use_bias=True, 
                             kernel_parameterizer=None,
                             activation=tf.nn.leaky_relu),
            tfc.SignalConv2D(self.num_filters, 
                             (5, 5), 
                             name="layer_1", 
                             corr=False, 
                             strides_up=2, 
                             padding=self.padding, 
                             use_bias=True, 
                             kernel_parameterizer=None,
                             activation=tf.nn.leaky_relu),
        ]
        
        # Note the linear activation
        self.loc_head = tfc.SignalConv2D(self.num_output_channels, 
                                         (3, 3), 
                                         name="layer_loc", 
                                         corr=False, 
                                         strides_up=1, 
                                         padding=self.padding, 
                                         use_bias=True, 
                                         kernel_parameterizer=None,
                                         activation=None)
        
        # Note the exponential activation
        self.scale_head = tfc.SignalConv2D(self.num_output_channels, 
                                           (3, 3), 
                                           name="layer_scale", 
                                           corr=False, 
                                           strides_up=1, 
                                           padding=self.padding, 
                                           use_bias=True, 
                                           kernel_parameterizer=None,
                                           activation=tf.math.exp)
        
        super(SynthesisTransform_2, self).build(input_shape)

        
    def call(self, inputs):
        
        activations = inputs
        
        # Omit the loc and scale heads
        for layer in self._layers[:-2]:
            activations = layer(activations)
            
        self.loc = self.loc_head(activations)
        self.scale = self.scale_head(activations)
        
        self.prior = tfd.Normal(loc=self.loc, scale=self.scale)
            
        return self.prior.sample()
    
    
# =====================================================================
# =====================================================================
# PLN definition
# =====================================================================
# =====================================================================

class ProbabilisticLadderNetwork(tfl.Layer):
    
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
            
        else:
            raise Exception("Unknown likelihood: {}".format(likelihood))
        
        super(ProbabilisticLadderNetwork, self).__init__(*args, **kwargs)
        
    @property
    def first_level_kl(self):
        return tfd.kl_divergence(self.posterior_1, self.prior_1)
        
        
    @property
    def second_level_kl(self):
        return tfd.kl_divergence(self.posterior_2, self.prior_2)    
        
        
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
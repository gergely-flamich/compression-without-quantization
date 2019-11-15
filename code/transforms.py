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
        
        # Note the tf.nn.softplus activation
        self.scale_head = tfc.SignalConv2D(filters=self.num_latent_channels, 
                                           kernel_support=(5, 5), 
                                           name="layer_scale", 
                                           corr=True, 
                                           strides_down=2,
                                           padding=self.padding, 
                                           use_bias=False,
                                           activation=tf.nn.softplus)
        
        super(AnalysisTransform_1, self).build(input_shape)
          
            
    def call(self, activations):
        
        # Omit the loc and scale heads
        for layer in self.layers:
            activations = layer(activations)
            
        self.loc = self.loc_head(activations)
        self.scale = self.scale_head(activations)
        
        self.posterior = tfd.Normal(loc=self.loc,
                                    scale=self.scale)

        # TODO why is this N(0, 1)
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
        self.num_latent_channels = num_latent_channels
        
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
        self.loc_head = tfc.SignalConv2D(filters=self.num_latent_channels, 
                                         kernel_support=(5, 5),
                                         name="layer_loc",
                                         corr=True,
                                         strides_down=2,
                                         padding=self.padding,
                                         use_bias=False,
                                         activation=None)
        
        # Note the tf.nn.softplus activation
        self.scale_head = tfc.SignalConv2D(filters=self.num_latent_channels, 
                                           kernel_support=(5, 5),
                                           name="layer_scale",
                                           corr=True,
                                           strides_down=2,
                                           padding=self.padding,
                                           use_bias=False,
                                           activation=tf.nn.softplus)
        
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
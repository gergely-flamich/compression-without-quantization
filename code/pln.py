import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
import tenosrflow_probability as tfp
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
        
        super(AnalysisTransorm_1).__init__(*args, **kwargs)
        
        
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
        self._loc_head = tfc.SignalConv2D(self.num_latent_channels, 
                                          (5, 5), 
                                          name="layer_loc", 
                                          corr=True, 
                                          strides_down=2,
                                          padding=self.padding, 
                                          use_bias=True,
                                          activation=None)
        
        # Note the exponential activation
        self._scale_head = tfc.SignalConv2D(self.num_latent_channels, 
                                            (5, 5), 
                                            name="layer_scale", 
                                            corr=True, 
                                            strides_down=2,
                                            padding=self.padding, 
                                            use_bias=True,
                                            activation=tf.math.exp)
        
        super(AnalysisTransorm_1, self).build(input_shape)
          
            
    def call(self, inputs):
        
        activations = inputs
        
        for layer in self._layers:
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
                                         activation=None),
        
        # Note the sigmoid activation
        self.scale_head = tfc.SignalConv2D(self.num_filters, 
                                           (5, 5),
                                           name="layer_scale",
                                           corr=True,
                                           strides_down=2,
                                           padding=self.padding,
                                           use_bias=False,
                                           activation=tf.nn.sigmoid),
        
        super(AnalysisTransform_2, self).build(input_shape)

        
    def call(self, inputs):
        
        activations = inputs
        
        for layer in self._layers:
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
                                         activation=None),
        
        # Note the exponential activation
        self.scale_head = tfc.SignalConv2D(self.num_output_channels, 
                                           (3, 3), 
                                           name="layer_scale", 
                                           corr=False, 
                                           strides_up=1, 
                                           padding=self.padding, 
                                           use_bias=True, 
                                           kernel_parameterizer=None,
                                           activation=tf.math.exp),
        
        super(HyperSynthesisTransform, self).build(input_shape)

        
    def call(self, inputs):
        
        activations = inputs
        
        for layer in self._layers:
            activations = layer(activations)
            
        self.loc = self.loc_head(activations)
        self.scale = self.scale_head(activations)
        
        self.prior = tfd.Normal(loc=self.loc, scale=self.scale)
            
    return self.prior.sample()
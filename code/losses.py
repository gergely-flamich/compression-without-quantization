import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class SsimMultiscalePseudo():
    
    def __init__(self, loc, scale):
        
        self.loc = loc
        self.scale = scale
        
    def log_prob(self, inputs):
        
        return (1. / self.scale) * tf.image.ssim_multiscale(self.loc, inputs, max_val=1.0)
    
    
class MixPseudo():
    
    def __init__(self, loc, scale, alpha=0.84):
        
        self.loc = loc
        self.scale = scale
        self.alpha = alpha
        
        self.laplace = tfd.Laplace(loc=loc, scale=scale)
        self.ssim = SsimMultiscalePseudo(loc=loc, scale=scale)
        
    def log_prob(self, inputs):
        
        return self.alpha * self.ssim.log_prob(inputs) + (1 - self.alpha) * tf.reduce_mean(self.laplace.log_prob(inputs))
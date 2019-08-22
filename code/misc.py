import tensorflow.compat.v1 as tf

def stateless_normal_sample(loc, scale, num_samples, seed):
    
    rank = tf.rank(loc)
    sample_shape = tf.concat(([num_samples], tf.shape(loc)), axis=0)
    tile_coefs = tf.concat(([num_samples], tf.tile([1], [rank])), axis=0)
    
    # Draw 0 mean, 1 variance samples
    samples = tf.random.stateless_normal(shape=sample_shape, 
                                         seed=tf.concat(([seed], [42]), axis=0))
    
    # Transform them to the right thing by scaling and translating appropriately
    samples = tf.tile(tf.expand_dims(scale, 0), tile_coefs) * samples
    samples = tf.tile(tf.expand_dims(loc, 0), tile_coefs) + samples
    
    return samples
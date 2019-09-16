# ==============================================================================
# Imports
# ==============================================================================
import numpy as np

import os, glob
from tqdm import tqdm as tqdm

import tensorflow.compat.v1 as tf
tfq = tf.quantization

import tensorflow_probability as tfp
tfd = tfp.distributions

from binary_io import to_bit_string, from_bit_string

from misc import stateless_normal_sample

# ==============================================================================================
# ==============================================================================================
# ==============================================================================================
#
# Importance Sampling
#
# ==============================================================================================
# ==============================================================================================
# ==============================================================================================

def code_importance_sample(t_loc,
                            t_scale,
                            p_loc,
                            p_scale,
                            n_coding_bits,
                            seed):
    
    
    target=tfd.Normal(loc=t_loc,
                      scale=t_scale)

    proposal=tfd.Normal(loc=p_loc,
                        scale=p_scale)
    
    #print("Taking {} samples per step".format(n_samples))
    
    sample_index = []
    
    kls = tfd.kl_divergence(target, proposal)
    total_kl = tf.reduce_sum(kls)
    
    num_samples = tf.cast(tf.math.ceil(tf.exp(total_kl)), tf.int32)
    
    # Set new seed
    #samples = proposal.sample(num_samples, seed=seed) 
    samples = stateless_normal_sample(loc=p_loc, 
                                      scale=p_scale, 
                                      num_samples=num_samples, 
                                      seed=seed)

    importance_weights = tf.reduce_sum(target.log_prob(samples) - proposal.log_prob(samples), axis=1)

    index = tf.argmax(importance_weights)
    best_sample = samples[index:index + 1, :]
    
    #index, best_sample = sess.run([idx, best_samp])
    
#     if np.log(index + 1) / np.log(2) > n_coding_bits:
#         raise Exception("Not enough bits to code importance sample!")
    
    # Turn the index into a bitstring
    bitcode = tf.numpy_function(to_bit_string, [index, n_coding_bits], tf.string)

    return best_sample, bitcode


def decode_importance_sample(sample_index, 
                              p_loc,
                              p_scale,
                              seed):

    index = tf.numpy_function(from_bit_string, [sample_index], tf.int64)
    
    samples = stateless_normal_sample(loc=p_loc,
                                      scale=p_scale,
                                      num_samples=tf.cast(index, tf.int32) + 1,
                                      seed=seed)
    
    return samples[index:, ...]


def code_grouped_importance_sample(sess,
                                    target, 
                                    proposal, 
                                    seed,
                                    n_bits_per_group,
                                    max_group_size_bits=4,
                                    dim_kl_bit_limit=12,
                                    return_group_indices_only=False):
    
    # Make sure the distributions have the correct type
    if target.dtype is not tf.float32:
        raise Exception("Target datatype must be float32!")
        
    if proposal.dtype is not tf.float32:
        raise Exception("Proposal datatype must be float32!")
        
        
    num_dimensions = sess.run(tf.reduce_prod(tf.shape(proposal.loc)))
    
    # rescale proposal by the proposal
    p_loc = sess.run(tf.reshape(tf.zeros_like(proposal.loc), [-1]))
    p_scale = sess.run(tf.reshape(tf.ones_like(proposal.scale), [-1]))
    
    # rescale target by the proposal
    t_loc = tf.reshape((target.loc - proposal.loc) / proposal.scale, [-1])
    t_scale = tf.reshape(target.scale / proposal.scale, [-1])
    
    # If we're going to do importance sampling, separate out dimensions with large KL,
    # we'll deal with them separately.
    kl_bits = tf.reshape(tfd.kl_divergence(target, proposal), [-1]) / np.log(2)

    t_loc = sess.run(tf.where(kl_bits <= dim_kl_bit_limit, t_loc, p_loc))
    t_scale = sess.run(tf.where(kl_bits <= dim_kl_bit_limit, t_scale, p_scale))

    # We'll send the quantized samples for dimensions with high KL
    outlier_indices = tf.where(kl_bits > dim_kl_bit_limit)

    target_samples = tf.reshape(target.sample(), [-1])

    # Select only the bits of the sample that are relevant
    outlier_samples = tf.gather_nd(target_samples, outlier_indices)

    # Halve precision
    outlier_samples = tfq.quantize(outlier_samples, -30, 30, tf.quint16).output

    outlier_extras = (tf.reshape(outlier_indices, [-1]), outlier_samples)
    
    kl_divergences = tf.reshape(
        tfd.kl_divergence(tfd.Normal(loc=t_loc, scale=t_scale), 
                          tfd.Normal(loc=p_loc, scale=p_scale)), [-1])

    kl_divs = sess.run(kl_divergences)
    group_start_indices = [0]
    group_kls = []

    total_kl_bits = np.sum(kl_divs) / np.log(2)

    print("Total KL to split up: {:.2f} bits, "
          "maximum bits per group: {}, "
          "estimated number of groups: {},"
          "coding {} dimensions".format(total_kl_bits, 
                                        n_bits_per_group,
                                        total_kl_bits // n_bits_per_group + 1,
                                        num_dimensions
                                        ))

    current_group_size = 0
    current_group_kl = 0
    
    n_nats_per_group = n_bits_per_group * np.log(2) - 1

    for idx in range(num_dimensions):

        group_bits = np.log(current_group_size + 1) / np.log(2)
        
        if group_bits > max_group_size_bits or \
           current_group_kl + kl_divs[idx] > n_nats_per_group or \
           idx == num_dimensions - 1:

            group_start_indices.append(idx)
            group_kls.append(current_group_kl / np.log(2))

            current_group_size = 1
            current_group_kl = kl_divs[idx]
            
        else:
            current_group_kl += kl_divs[idx]
            current_group_size += 1
        
    print("Maximum group KL: {:.3f}".format(np.max(group_kls)))
    
    group_start_indices += [num_dimensions] 
    group_start_indices = np.array(group_start_indices)
    
    if return_group_indices_only:
        return group_start_indices, group_kls
    
    # ====================================================================== 
    # Sample each group
    # ====================================================================== 
    
    results = []
    
    # Get the importance sampling op before looping it to avoid graph construction cost
    # The length is variable, hence the shape is [None]
    target_loc = tf.placeholder(tf.float32, shape=[None])
    target_scale = tf.placeholder(tf.float32, shape=[None])
    
    prop_loc = tf.placeholder(tf.float32, shape=[None])
    prop_scale = tf.placeholder(tf.float32, shape=[None])
    
    seed_feed = tf.placeholder(tf.int32)

    result_ops = code_importance_sample(t_loc=target_loc,
                                         t_scale=target_scale,
                                         p_loc=prop_loc,
                                         p_scale=prop_scale,
                                         seed=seed_feed,
                                         n_coding_bits=n_bits_per_group)
            
    for i in tqdm(range(len(group_start_indices) - 1)):
        
        start_idx = group_start_indices[i]
        end_idx = group_start_indices[i + 1]
        
        
        result = sess.run(result_ops, feed_dict={target_loc: t_loc[start_idx:end_idx],
                                                 target_scale: t_scale[start_idx:end_idx],
                                                 prop_loc: p_loc[start_idx:end_idx],
                                                 prop_scale: p_scale[start_idx:end_idx],
                                                 seed_feed: seed + i
                                                })
        results.append(result)
        
    samples, codes = zip(*results)
    
    bitcode = tf.numpy_function(lambda code_words: ''.join([cw.decode("utf-8") for cw in code_words]), 
                                [codes], 
                                tf.string)
    
    sample = tf.concat(samples, axis=1)
    
    # Rescale the sample
    sample = tf.reshape(proposal.scale, [-1]) * sample + tf.reshape(proposal.loc, [-1])
    
    sample = tf.where(kl_bits <= dim_kl_bit_limit, tf.squeeze(sample), target_samples)
    
    sample, bitcode, outlier_extras = sess.run([sample, bitcode, outlier_extras])
    
    return sample, bitcode, group_start_indices, outlier_extras


def decode_grouped_importance_sample(sess,
                                     bitcode, 
                                     group_start_indices,
                                     proposal, 
                                     n_bits_per_group,
                                     seed,
                                     outlier_indices,
                                     outlier_samples):
    
    # Make sure the distributions have the correct type
    if proposal.dtype is not tf.float32:
        raise Exception("Proposal datatype must be float32!")
    
    num_dimensions = sess.run(tf.reduce_prod(tf.shape(proposal.loc)))
    
    # ====================================================================== 
    # Decode each group
    # ====================================================================== 
                
    samples = []
    
    group_start_indices += [num_dimensions]
    
    p_loc = sess.run(tf.reshape(tf.zeros_like(proposal.loc), [-1]))
    p_scale = sess.run(tf.reshape(tf.ones_like(proposal.scale), [-1]))

    # Placeholders
    sample_index = tf.placeholder(tf.string)
    
    prop_loc = tf.placeholder(tf.float32, shape=[None])
    prop_scale = tf.placeholder(tf.float32, shape=[None])
    
    seed_feed = tf.placeholder(tf.int32)
    
    # Get decoding op
    decode_op = decode_importance_sample(sample_index=sample_index,
                                          p_loc=prop_loc,
                                          p_scale=prop_scale,
                                          seed=seed_feed)

    for i in tqdm(range(len(group_start_indices) - 1)):
        
        samp = sess.run(decode_op, feed_dict = {
            sample_index: bitcode[n_bits_per_group * i: n_bits_per_group * (i + 1)],
            prop_loc: p_loc[group_start_indices[i]:group_start_indices[i + 1]],
            prop_scale: p_scale[group_start_indices[i]:group_start_indices[i + 1]],
            seed_feed: seed + i
        })
        
        samples.append(samp)

    sample = tf.concat(samples, axis=1)
    
    # Rescale the sample
    sample = tf.reshape(proposal.scale, [-1]) * sample + tf.reshape(proposal.loc, [-1])
    sample = tf.squeeze(sample)
    
    # Dequantize outliers
    outlier_samples = tfq.dequantize(tf.cast(outlier_samples, tf.quint16), -30, 30)
    
    # Add back the quantized outliers
    outlier_indices = tf.cast(tf.reshape(outlier_indices, [-1, 1]), tf.int32)
    outlier_samples = tf.reshape(outlier_samples, [-1])
    
    updates = tf.scatter_nd(outlier_indices, 
                            outlier_samples, 
                            sample.shape.as_list())
                            
    sample = tf.where(tf.equal(updates, 0), sample, updates)
    
    return sess.run(sample)
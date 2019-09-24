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

from binary_io import to_bit_string, from_bit_string, elias_delta_code, elias_delta_decode

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
                            seed,
                          return_index_only=False):
    
    
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
    #bitcode = tf.numpy_function(to_bit_string, [index, n_coding_bits], tf.string)
    
    if return_index_only:
        return best_sample, index + 1
    
    else:
        bitcode = tf.numpy_function(elias_delta_code, [index + 1], tf.string)

        return best_sample, bitcode


def decode_importance_sample(sample_index, 
                              p_loc,
                              p_scale,
                              seed,
                              use_index=False):

    if use_index:
        index = sample_index - 1
        
        samples = stateless_normal_sample(loc=p_loc,
                                      scale=p_scale,
                                      num_samples=tf.cast(index, tf.int32) + 1,
                                      seed=seed)
        
        return samples[-1:, ...]
    
    else:
        
        index, code_length = tf.numpy_function(elias_delta_decode, [sample_index], (tf.int64, tf.int64))

        index = index - 1

        samples = stateless_normal_sample(loc=p_loc,
                                          scale=p_scale,
                                          num_samples=tf.cast(index, tf.int32) + 1,
                                          seed=seed)

        return samples[-1:, ...], code_length, index, samples


def code_grouped_importance_sample(sess,
                                    target, 
                                    proposal, 
                                    seed,
                                    n_bits_per_group,
                                    max_group_size_bits=4,
                                    dim_kl_bit_limit=12,
                                    return_group_indices_only=False,
                                    return_indices=False,
                                   return_indices_only=False):
    
    # Make sure the distributions have the correct type
    if target.dtype is not tf.float32:
        raise Exception("Target datatype must be float32!")
        
    if proposal.dtype is not tf.float32:
        raise Exception("Proposal datatype must be float32!")
        
    num_dimensions = sess.run(tf.reduce_prod(tf.shape(proposal.loc)))
    
    # rescale proposal by the proposal
    p_loc = sess.run(tf.zeros_like(proposal.loc))
    p_scale = sess.run(tf.ones_like(proposal.scale))
    
    # rescale target by the proposal
    t_loc = (target.loc - proposal.loc) / proposal.scale
    t_scale = target.scale / proposal.scale
    
    # If we're going to do importance sampling, separate out dimensions with large KL,
    # we'll deal with them separately.
    kl_bits = tfd.kl_divergence(target, proposal) / np.log(2)

    t_loc = sess.run(tf.where(kl_bits <= dim_kl_bit_limit, t_loc, p_loc))
    t_scale = sess.run(tf.where(kl_bits <= dim_kl_bit_limit, t_scale, p_scale))

    # We'll send the quantized samples for dimensions with high KL
    outlier_indices = tf.where(kl_bits > dim_kl_bit_limit)

    target_samples = target.sample()

    # Select only the bits of the sample that are relevant
    outlier_samples = tf.gather_nd(target_samples, outlier_indices)

    # Halve precision
    outlier_samples = tfq.quantize(outlier_samples, -30, 30, tf.quint16).output

    outlier_extras = (tf.reshape(outlier_indices, [-1]), outlier_samples)
    
    kl_divergences = tfd.kl_divergence(tfd.Normal(loc=t_loc, scale=t_scale), 
                                       tfd.Normal(loc=p_loc, scale=p_scale))

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
                                         n_coding_bits=n_bits_per_group,
                                         return_index_only=return_indices_only or return_indices)
            
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
        
    # To build probability distribution we return the indices only
    if return_indices_only:
        samples, indices = zip(*results)
        return indices
    
    if return_indices:
        samples, indices = zip(*results)
        
    else:
        samples, codes = zip(*results)
    
        bitcode = tf.numpy_function(lambda code_words: ''.join([cw.decode("utf-8") for cw in code_words]), 
                                    [codes], 
                                    tf.string)
    
    sample = tf.concat(samples, axis=1)
    
    # Rescale the sample
    sample = proposal.scale * sample + proposal.loc
    
    sample = tf.where(kl_bits <= dim_kl_bit_limit, tf.squeeze(sample), target_samples)
    
    if return_indices:
        sample, outlier_extras = sess.run([sample, outlier_extras])
        return sample, indices, group_start_indices, outlier_extras
    else:
        sample, bitcode, outlier_extras = sess.run([sample, bitcode, outlier_extras])
        return sample, bitcode, group_start_indices, outlier_extras


def decode_grouped_importance_sample(sess,
                                     bitcode, 
                                     group_start_indices,
                                     proposal, 
                                     n_bits_per_group,
                                     seed,
                                     outlier_indices,
                                     outlier_samples,
                                     use_indices=False):
    
    # Make sure the distributions have the correct type
    if proposal.dtype is not tf.float32:
        raise Exception("Proposal datatype must be float32!")
    
    num_dimensions = sess.run(tf.reduce_prod(tf.shape(proposal.loc)))
    
    # The num dimensions is not communicated, so appending it here
    group_start_indices.append(num_dimensions)

    # ====================================================================== 
    # Decode each group
    # ====================================================================== 
                
    samples = []
    
    #group_start_indices += [num_dimensions]
    
    p_loc = sess.run(tf.zeros_like(proposal.loc))
    p_scale = sess.run(tf.ones_like(proposal.scale))

    # Placeholders
    if use_indices:
        sample_index = tf.placeholder(tf.int32)
    else:
        sample_index = tf.placeholder(tf.string)
    
    prop_loc = tf.placeholder(tf.float32, shape=[None])
    prop_scale = tf.placeholder(tf.float32, shape=[None])
    
    seed_feed = tf.placeholder(tf.int32)
    
    # Get decoding op
    decode_op = decode_importance_sample(sample_index=sample_index,
                                          p_loc=prop_loc,
                                          p_scale=prop_scale,
                                          use_index=use_indices,
                                          seed=seed_feed)

    for i in tqdm(range(len(group_start_indices) - 1)):
        
        samp, codelength, index, ss = sess.run(decode_op, feed_dict = {
            sample_index: bitcode[i] if use_indices else bitcode,
            prop_loc: p_loc[group_start_indices[i]:group_start_indices[i + 1]],
            prop_scale: p_scale[group_start_indices[i]:group_start_indices[i + 1]],
            seed_feed: seed + i
        })
        
        if not use_indices:
            # Cut the code to only the relevant bits
            bitcode = bitcode[codelength:]
        
        samples.append(samp)
        
#         print(index)
#         print(samp)
#         print(ss)

    sample = tf.concat(samples, axis=1)
    
    # Rescale the sample
    sample = proposal.scale * sample + proposal.loc
    sample = tf.squeeze(sample)
    
    # Dequantize outliers
    outlier_samples = tfq.dequantize(tf.cast(outlier_samples, tf.quint16), -30, 30)
    
    # Add back the quantized outliers
    outlier_indices = tf.cast(tf.reshape(outlier_indices, [-1, 1]), tf.int32)
    outlier_samples = outlier_samples
    
    updates = tf.scatter_nd(outlier_indices, 
                            outlier_samples, 
                            sample.shape.as_list())
                            
    sample = tf.where(tf.equal(updates, 0), sample, updates)
    
    return sess.run(sample)
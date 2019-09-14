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
# Greedy Sampling
#
# ==============================================================================================
# ==============================================================================================
# ==============================================================================================

def code_greedy_sample(t_loc, 
                        t_scale,
                        p_loc,
                        p_scale,
                        n_bits_per_step, 
                        n_steps, 
                        seed, 
                        rho=1.):
    
    n_samples = int(2**n_bits_per_step)

    # The scale divisor needs to be square rooted because
    # we are dealing with standard deviations and not variances
    scale_divisor = np.sqrt(n_steps)
    
    proposal_shard = tfd.Normal(loc=p_loc / n_steps,
                                scale=rho * p_scale / scale_divisor)
    
    target = tfd.Normal(loc=t_loc,
                        scale=t_scale)

    # Setup greedy sampler for loop
    def loop_step(i, sample_index, best_sample):
        samples = stateless_normal_sample(loc=proposal_shard.loc, 
                                          scale=proposal_shard.scale, 
                                          num_samples=n_samples, 
                                          seed=1000 * seed + i)
        
        test_samples = tf.tile(tf.expand_dims(best_sample, 0), [n_samples, 1]) + samples

        log_probs = tf.reduce_sum(target.log_prob(test_samples), axis=1)

        index = tf.argmax(log_probs)

        best_sample = test_samples[index, :]

        return [i + 1, tf.concat((sample_index, [index]), axis=0), best_sample]
    
    i = tf.constant(0)
    best_sample = tf.zeros(tf.shape(p_loc), dtype=tf.float32)
    sample_index = tf.cast([], tf.int32)
    
    cond = lambda i, sample_index, best_sample: i < n_steps

    _, sample_index, best_sample = tf.while_loop(cond=cond,
                                   body=loop_step, 
                                   loop_vars=[i, sample_index, best_sample],
                                   shape_invariants=[i.get_shape(), 
                                                     tf.TensorShape([None]), 
                                                     best_sample.get_shape()])
    
    
    sample_index = tf.map_fn(lambda x: tf.numpy_function(to_bit_string, [x, n_bits_per_step], tf.string), 
                             sample_index,
                             dtype=tf.string)
    
    sample_index = tf.numpy_function(lambda indices: ''.join([ind.decode('utf-8') for ind in indices]),
                                     [sample_index],
                                     tf.string)
    
    return best_sample, sample_index



def decode_greedy_sample(sample_index, 
                          p_loc,
                          p_scale,
                          n_bits_per_step, 
                          n_steps, 
                          seed, 
                          rho=1.):
    
    
    # Perform a for loop for the below list comprehension
    #
    #     indices = [from_bit_string(sample_index[i:i + n_bits_per_step]) 
    #                for i in range(0, n_bits_per_step * n_steps, n_bits_per_step)]
    #
    i = tf.constant(0, tf.int32)
    indices = tf.cast([], tf.int32)
    
    cond = lambda i, indices: i < n_bits_per_step * n_steps

    def index_loop_step(i, indices):
        
        index = tf.numpy_function(from_bit_string, 
                                  [tf.strings.substr(sample_index, i, n_bits_per_step)], 
                                  tf.int64)
        
        index = tf.cast(index, tf.int32)
        
        return [i + n_bits_per_step, tf.concat((indices, [index]), axis=0)]
     
    _, indices = tf.while_loop(cond=cond,
                               body=index_loop_step, 
                               loop_vars=[i, indices],
                               shape_invariants=[i.get_shape(), 
                                                 tf.TensorShape([None])])
    
    # ---------------------------------------------------------------------
    # Reconver the sample
    # ---------------------------------------------------------------------
    
    # The scale divisor needs to be square rooted because
    # we are dealing with standard deviations and not variances
    scale_divisor = np.sqrt(n_steps)    
    
    proposal_shard = tfd.Normal(loc=p_loc / n_steps,
                                scale=rho * p_scale / scale_divisor)    
    
    n_samples = int(2**n_bits_per_step)
    
    # Loop variables
    i = tf.constant(0, tf.int32)
    sample = tf.zeros(tf.shape(p_loc), dtype=tf.float32)
    
    # Loop condition
    cond = lambda i, indices: i < n_steps

    # Loop body
    def sample_loop_step(i, sample):
        
        samples = tf.tile(tf.expand_dims(sample, 0), [n_samples, 1])
        
        samples = samples + stateless_normal_sample(loc=proposal_shard.loc, 
                                                    scale=proposal_shard.scale, 
                                                    num_samples=n_samples, 
                                                    seed=1000 * seed + i)

        return [i + 1, samples[indices[i], :]]
     
    # Run loop
    _, sample = tf.while_loop(cond=cond,
                              body=sample_loop_step, 
                              loop_vars=[i, sample],
                              shape_invariants=[i.get_shape(), 
                                                sample.get_shape()])
    
    return sample


def code_grouped_greedy_sample(sess,
                                target, 
                               proposal,
                               n_steps, 
                               n_bits_per_step,
                               seed,
                               max_group_size_bits=12,
                               adaptive=True,
                               backfitting_steps=0,
                               use_log_prob=False,
                               rho=1.):
    
    # Make sure the distributions have the correct type
    if target.dtype is not tf.float32:
        raise Exception("Target datatype must be float32!")
        
    if proposal.dtype is not tf.float32:
        raise Exception("Proposal datatype must be float32!")
    
    n_bits_per_group = n_bits_per_step * n_steps
    
    num_dimensions = sess.run(tf.reduce_prod(tf.shape(proposal.loc)))
    
    # rescale proposal by the proposal
    p_loc = sess.run(tf.reshape(tf.zeros_like(proposal.loc), [-1]))
    p_scale = sess.run(tf.reshape(tf.ones_like(proposal.scale), [-1]))
    
    # rescale target by the proposal
    t_loc = sess.run(tf.reshape((target.loc - proposal.loc) / proposal.scale, [-1]))
    t_scale = sess.run(tf.reshape(target.scale / proposal.scale, [-1]))
    
    kl_divergences = tf.reshape(tfd.kl_divergence(target, proposal), [-1])
        
    # ====================================================================== 
    # Preprocessing step: determine groups for sampling
    # ====================================================================== 

    group_start_indices = [0]
    group_kls = []
    
    kl_divs = sess.run(kl_divergences)

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
        
        if group_bits >= max_group_size_bits or \
           current_group_kl + kl_divs[idx] >= n_nats_per_group or \
           idx == num_dimensions - 1:

            group_start_indices.append(idx)
            group_kls.append(current_group_kl / np.log(2))

            current_group_size = 1
            current_group_kl = kl_divs[idx]
            
        else:
            current_group_kl += kl_divs[idx]
            current_group_size += 1
            
    # ====================================================================== 
    # Sample each group
    # ====================================================================== 
    
    results = []
    
    group_start_indices += [num_dimensions] 
    
    # Get the importance sampling op before looping it to avoid graph construction cost
    # The length is variable, hence the shape is [None]
    target_loc = tf.placeholder(tf.float32, shape=[None])
    target_scale = tf.placeholder(tf.float32, shape=[None])
    
    prop_loc = tf.placeholder(tf.float32, shape=[None])
    prop_scale = tf.placeholder(tf.float32, shape=[None])
    
    seed_feed = tf.placeholder(tf.int32)
    
    greedy_op = code_greedy_sample(t_loc=target_loc,
                                    t_scale=target_scale,
                                    p_loc=prop_loc,
                                    p_scale=prop_scale,
                                    n_bits_per_step=n_bits_per_step,
                                    n_steps=n_steps, 
                                    seed=seed_feed,
                                    rho=rho)
    
    for i in tqdm(range(len(group_start_indices) - 1)):
        
        start_idx = group_start_indices[i]
        end_idx = group_start_indices[i + 1]
        
        result = sess.run(greedy_op, feed_dict={target_loc: t_loc[start_idx:end_idx],
                                                target_scale: t_scale[start_idx:end_idx],
                                                prop_loc: p_loc[start_idx:end_idx],
                                                prop_scale: p_scale[start_idx:end_idx],
                                                seed_feed: seed + i})
        
        results.append(result)
        
    samples, codes = zip(*results)
    
    bitcode = ''.join([c.decode('utf-8') for c in codes])
    sample = tf.concat(samples, axis=0)
    
    # Rescale the sample
    sample = tf.reshape(proposal.scale, [-1]) * sample + tf.reshape(proposal.loc, [-1])
    
    sample = sess.run(sample)
    
    return sample, bitcode, group_start_indices
  
    
def decode_grouped_greedy_sample(sess,
                                  bitcode, 
                                 group_start_indices,
                                 proposal, 
                                 n_bits_per_step, 
                                 n_steps, 
                                 seed,
                                 adaptive=True,
                                 rho=1.):
    
    # Make sure the distributions have the correct type
    if proposal.dtype is not tf.float32:
        raise Exception("Proposal datatype must be float32!")
    
    n_bits_per_group = n_bits_per_step * n_steps
    
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
    decode_greedy_op = decode_greedy_sample(sample_index=sample_index, 
                                             p_loc=prop_loc,
                                             p_scale=prop_scale,
                                             n_bits_per_step=n_bits_per_step, 
                                             n_steps=n_steps, 
                                             seed=seed_feed, 
                                             rho=rho)

    for i in tqdm(range(len(group_start_indices) - 1)):
        if bitcode[n_bits_per_group * i: n_bits_per_group * (i + 1)] == '':
            break
        
        samp = sess.run(decode_greedy_op, feed_dict = {
            sample_index: bitcode[n_bits_per_group * i: n_bits_per_group * (i + 1)],
            prop_loc: p_loc[group_start_indices[i]:group_start_indices[i + 1]],
            prop_scale: p_scale[group_start_indices[i]:group_start_indices[i + 1]],
            seed_feed: seed + i
        })
        
        samples.append(samp)
    
        
    sample = tf.concat(samples, axis=0)
    
    # Rescale the sample
    sample = tf.reshape(proposal.scale, [-1]) * sample + tf.reshape(proposal.loc, [-1])
    
    return sess.run(sample)
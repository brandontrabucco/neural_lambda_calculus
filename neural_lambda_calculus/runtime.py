"""Author: Brandon Trabucco, Kavi Gupta, Copyright 2019"""


import neural_lambda_calculus.module as module
import neural_lambda_calculus.hyper_network as hyper_network
import neural_lambda_calculus.transformer as transformer
import tensorflow as tf
import numpy as np


class Runtime(module.Module):

    def __init__(self, maximum_depth, hidden_size, num_layers, output_shapes, num_heads, hidden_sizes):
        self.hyper_network = hyper_network.HyperNetwork(hidden_size, num_layers, output_shapes)
        self.transformer = transformer.Transformer(num_heads, hidden_sizes)
        self.maximum_depth = maximum_depth

    def build_program(self, seed):
        model_weights = self.hyper_network(seed)
        Q_ws, K_ws, V_ws, S_ws = model_weights[0::8], model_weights[1::8], model_weights[2::8], model_weights[3::8]
        H_ws, H_bs, F_ws, F_bs = model_weights[4::8], model_weights[5::8], model_weights[6::8], model_weights[7::8]
        G_w, G_b = model_weights[-2], model_weights[-1]
        return Q_ws, K_ws, V_ws, S_ws, H_ws, H_bs, F_ws, F_bs, G_w, G_b

    def execute_program(self, seed, environment, all_probs, current_depth):
        """Performs a step of neural lambda calculus using a seed and environment.
        Args:
            seed:           a float32 tensor of shape [batch_size, num_features]
            environment:    a float32 tensor of shape [batch_size, bank_size, num_features]
            all_probs:      a float32 tensor of shape [batch_size, num_actions]
            current_depth:  an int that indicates the current level of recursion
        Returns: 
            the resulting next_seed tensor."""
        # First execute the current program on the environment
        result = self.transformer(environment, *self.build_program(seed))
        # Then collect the condition probability
        condition, rest = result[:, 0], result[:, 1:]
        condition = tf.sigmoid(condition)
        random_samples = tf.random_uniform([tf.shape(condition)[0]])
        mask = random_samples < condition
        # Compute the probability of the state we ended up in
        action_probs = tf.where(mask, condition, 1.0 - condition)
        all_probs = tf.concat([all_probs, tf.expand_dims(action_probs, 1)], 1)
        # Take the left and right seeds and evaluate them
        left, right = tf.split(rest, 2, axis=1)
        left_result, left_probs = self.execute_program(left, environment, all_probs, current_depth + 1)
        right_result, right_probs = self.execute_program(right, environment, all_probs, current_depth + 1)
        # Compute additional probabilities from left and right subtrees
        residual_left_probs = left_probs[:, tf.shape(all_probs)[1]:]
        residual_right_probs = right_probs[:, tf.shape(all_probs)[1]:]
        padded_probs = tf.pad(all_probs, [[0, 0], [0, 
            tf.shape(residual_left_probs)[1] + tf.shape(residual_right_probs)[1]]])
        joined_probs = tf.concat([all_probs, residual_left_probs, residual_right_probs], 1)
        # Compute the evaluation of program left on right
        joined_result, joined_probs = self.execute_program(left_result,
            tf.concat([tf.expand_dims(right_result, 1), environment], 1), joined_probs, current_depth + 1)
        # Merge the results according to the mask
        next_seed = tf.where(mask, right, joined_result)
        next_probs = tf.where(mask, padded_probs, joined_probs)
        return next_seed, next_probs

    def __call__(self, seed, environment):
        return self.execute_program(seed, environment, tf.zeros([tf.shape(seed)[0], 0]), 0)

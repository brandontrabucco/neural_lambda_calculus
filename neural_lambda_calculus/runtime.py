"""Author: Brandon Trabucco, Kavi Gupta, Copyright 2019"""


import neural_lambda_calculus.module as module
import neural_lambda_calculus.transformer as transformer
import tensorflow as tf
from collections import defaultdict


class Runtime(module.Module):

    def __init__(self, max_call_depth, max_branch_depth, num_heads, attention_hidden_size, 
            dense_hidden_size, num_layers, hidden_size, output_size):
        self.transformer = transformer.Transformer(num_heads, attention_hidden_size, 
            dense_hidden_size, num_layers, hidden_size, output_size)
        self.max_call_depth = max_call_depth - 1
        self.max_branch_depth = max_branch_depth - 1
        self.num_endpoints = 0

    def execute_function(self, seed, environment):
        result = tf.squeeze(self.transformer(environment, tf.expand_dims(seed, 1)), 1)
        next_seeds, condition = result[:, 1:], tf.sigmoid(result[:, 0])
        return (*tf.split(next_seeds, 2, axis=1)), condition, (
            tf.random_uniform([tf.shape(condition)[0]]) < condition)

    def evaluate_symbols(self, left, right, environment, all_probs, call_depth, branch_depth, halt=False):
        return (left, all_probs, right, all_probs) if self.max_branch_depth == -1 else (
            *self.execute_program(left, environment, all_probs, call_depth + 1, branch_depth + 1, halt=halt), 
            *self.execute_program(right, environment, all_probs, call_depth + 1, branch_depth + 1, halt=halt)) 

    def apply_symbols(self, left_result, right_result, environment, left_probs, right_probs, all_probs, 
            call_depth, branch_depth, halt=False):
        residual_left_probs, residual_right_probs = (left_probs[:, tf.shape(all_probs)[1]:], 
            right_probs[:, tf.shape(all_probs)[1]:])
        joined_probs = tf.concat([all_probs, residual_left_probs, residual_right_probs], 1)
        joined_result, joined_probs = self.execute_program(left_result, tf.concat([tf.expand_dims(right_result, 1), 
            environment], 1), joined_probs, call_depth + 1, branch_depth + (1 if branch_depth > 0 else 0), halt=halt)
        padded_probs = tf.pad(all_probs, [[0, 0], [0, tf.shape(joined_probs)[1] - tf.shape(all_probs)[1]]])
        return joined_result, joined_probs, padded_probs

    def execute_program(self, seed, environment, all_probs, call_depth, branch_depth, halt=False):
        left, right, condition, mask = self.execute_function(seed, environment)
        if halt:
            self.num_endpoints += 1
            print("Building endpoint {} at depth {} and branch {}".format(
                self.num_endpoints, call_depth, branch_depth))
            return right, tf.concat([all_probs, tf.expand_dims(condition, 1)], 1)
        all_probs = tf.concat([all_probs, tf.expand_dims(tf.where(mask, condition, 1.0 - condition), 1)], 1)
        left_result, left_probs, right_result, right_probs = self.evaluate_symbols(
            left, right, environment, all_probs, call_depth, branch_depth, halt=((call_depth >= self.max_call_depth) or (
                branch_depth >= self.max_branch_depth)))
        joined_result, joined_probs, padded_probs = self.apply_symbols(left_result, right_result, environment, 
            left_probs, right_probs, all_probs, call_depth, branch_depth, halt=((call_depth >= self.max_call_depth) or (
                (branch_depth >= self.max_branch_depth) and (self.max_branch_depth > 0))))
        return tf.where(mask, right, joined_result), tf.where(mask, padded_probs, joined_probs)

    def __call__(self, seed, environment):
        return self.execute_program(seed, environment, tf.zeros([tf.shape(seed)[0], 0]), 0, 0, halt=(0 >= self.max_call_depth))

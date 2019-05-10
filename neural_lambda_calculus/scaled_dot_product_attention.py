"""Author: Brandon Trabucco, Kavi Gupta, Copyright 2019"""


import neural_lambda_calculus.module as module
import tensorflow as tf


class ScaledDotProductAttention(module.Module):

    def __init__(self, num_heads, hidden_size, output_size):
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.query_map = tf.keras.layers.Dense(hidden_size * num_heads, use_bias=False)
        self.key_map = tf.keras.layers.Dense(hidden_size * num_heads, use_bias=False)
        self.value_map = tf.keras.layers.Dense(hidden_size * num_heads, use_bias=False)
        self.output_map = tf.keras.layers.Dense(output_size, use_bias=False)
    
    def __call__(self, queries, keys, values, use_mask=True):
        batch_size, num_queries, sequence_length = tf.shape(queries)[0], tf.shape(queries)[1], tf.shape(values)[1]
        mask = tf.matrix_band_part(tf.ones([batch_size, self.num_heads, num_queries, sequence_length]), 0, -1)
        Q, K, V = self.query_map(queries), self.key_map(keys), self.value_map(values)
        Q = tf.transpose(tf.reshape(Q, [batch_size, num_queries, self.num_heads, self.hidden_size]), [0, 2, 1, 3])
        K = tf.transpose(tf.reshape(K, [batch_size, sequence_length, self.num_heads, self.hidden_size]), [0, 2, 1, 3])
        V = tf.transpose(tf.reshape(V, [batch_size, sequence_length, self.num_heads, self.hidden_size]), [0, 2, 1, 3])
        S = tf.matmul(((1.0 - mask) if use_mask else 1.0) * tf.nn.softmax(
            tf.matmul(Q, tf.transpose(K, [0, 1, 3, 2])) / tf.sqrt(float(self.hidden_size))), V)
        return self.output_map(tf.reshape(tf.transpose(
            S, [0, 2, 1, 3]), [batch_size, num_queries, self.num_heads * self.hidden_size]))
        
    def trainable_variables(self):
        layer_variables = (self.query_map.trainable_variables + self.key_map.trainable_variables + 
            self.value_map.trainable_variables + self.output_map.trainable_variables )
        return layer_variables
    
    def variables(self):
        layer_variables = (self.query_map.variables + self.key_map.variables + 
            self.value_map.variables + self.output_map.variables )
        return layer_variables
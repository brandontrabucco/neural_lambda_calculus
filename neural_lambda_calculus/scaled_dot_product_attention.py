"""Author: Brandon Trabucco, Kavi Gupta, Copyright 2019"""


import neural_lambda_calculus.module as module
import neural_lambda_calculus.dense as dense
import tensorflow as tf


class ScaledDotProductAttention(module.Module):

    def __init__(self, num_heads, hidden_size):
        self.num_heads = num_heads
        self.hidden_size = hidden_size
    
    def __call__(self, queries, keys, values, Q_w, K_w, V_w, S_w):
        batch_size, num_queries, sequence_length = (tf.shape(queries)[0], 
            tf.shape(queries)[1], tf.shape(values)[1])
        triangular = tf.matrix_band_part(tf.ones([batch_size, self.num_heads, 
            num_queries, sequence_length]), 0, -1)
        Q, K, V = dense.dense(queries, Q_w), dense.dense(keys, K_w), dense.dense(values, V_w)
        Q = tf.transpose(tf.reshape(Q, [batch_size, num_queries, self.num_heads, 
            self.hidden_size]), [0, 2, 1, 3])
        K = tf.transpose(tf.reshape(K, [batch_size, sequence_length, self.num_heads, 
            self.hidden_size]), [0, 2, 1, 3])
        V = tf.transpose(tf.reshape(V, [batch_size, sequence_length, self.num_heads, 
            self.hidden_size]), [0, 2, 1, 3])
        S = tf.matmul((1.0 - triangular) * tf.nn.softmax(
            tf.matmul(Q, tf.transpose(K, [0, 1, 3, 2])) / tf.sqrt(float(self.hidden_size))), V)
        return dense.dense(tf.reshape(tf.transpose(
            S, [0, 2, 1, 3]), [batch_size, num_queries, self.num_heads * self.hidden_size]), S_w)
        
    def trainable_variables(self):
        return []
    
    def variables(self):
        return []
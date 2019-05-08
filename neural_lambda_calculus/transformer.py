"""Author: Brandon Trabucco, Kavi Gupta, Copyright 2019"""


import neural_lambda_calculus.module as module
import neural_lambda_calculus.scaled_dot_product_attention as attention
import neural_lambda_calculus.dense as dense
import tensorflow as tf


class Transformer(module.Module):

    def __init__(self, num_heads, hidden_sizes):
        self.attention_layers = [attention.ScaledDotProductAttention(a, b) for a, b in zip(
            num_heads, hidden_sizes)]
    
    def __call__(self, x, Q_ws, K_ws, V_ws, S_ws, H_ws, H_bs, F_ws, F_bs, G_w, G_b):
        for (layer, Q_w, K_w, V_w, S_w, H_w, H_b, F_w, F_b) in zip(
                self.attention_layers, self.fc_hidden_layers, self.fc_output_layers):
            x = dense.dense(tf.nn.relu(dense.dense(
                layer(x, x, x, Q_w, K_w, V_w, S_w), H_w, H_b)), F_w, F_b)
        return dense.dense(tf.reduce_mean(x, [1]), G_w, G_b)
        
    def trainable_variables(self):
        return []
    
    def variables(self):
        return []
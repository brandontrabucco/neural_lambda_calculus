"""Author: Brandon Trabucco, Kavi Gupta, Copyright 2019"""


import neural_lambda_calculus.module as module
import neural_lambda_calculus.scaled_dot_product_attention as attention
import tensorflow as tf


class Transformer(module.Module):

    def __init__(self, num_heads, attention_hidden_size, dense_hidden_size, 
            num_layers, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.input_one_layer = tf.keras.layers.Dense(hidden_size, use_bias=False)
        self.input_two_layer = tf.keras.layers.Dense(hidden_size, use_bias=False)
        self.attention_one_layers = [attention.ScaledDotProductAttention(num_heads, 
            attention_hidden_size, hidden_size) for i in range(num_layers)]
        self.attention_one_norms = [tf.keras.layers.BatchNormalization(
            ) for i in range(num_layers)]
        self.dense_one_hidden_layers = [tf.keras.layers.Dense(
            dense_hidden_size) for i in range(num_layers)]
        self.dense_one_output_layers = [tf.keras.layers.Dense(
            hidden_size) for i in range(num_layers)]
        self.dense_one_norms = [tf.keras.layers.BatchNormalization(
            ) for i in range(num_layers)]
        self.attention_two_layers = [attention.ScaledDotProductAttention(num_heads, 
            attention_hidden_size, hidden_size) for i in range(num_layers)]
        self.attention_two_norms = [tf.keras.layers.BatchNormalization(
            ) for i in range(num_layers)]
        self.dense_two_hidden_layers = [tf.keras.layers.Dense(
            dense_hidden_size) for i in range(num_layers)]
        self.dense_two_output_layers = [tf.keras.layers.Dense(
            hidden_size) for i in range(num_layers)]
        self.dense_two_norms = [tf.keras.layers.BatchNormalization(
            ) for i in range(num_layers)]
        self.attention_three_layers = [attention.ScaledDotProductAttention(num_heads, 
            attention_hidden_size, hidden_size) for i in range(num_layers)]
        self.attention_three_norms = [tf.keras.layers.BatchNormalization(
            ) for i in range(num_layers)]
        self.dense_three_hidden_layers = [tf.keras.layers.Dense(
            dense_hidden_size) for i in range(num_layers)]
        self.dense_three_output_layers = [tf.keras.layers.Dense(
            hidden_size) for i in range(num_layers)]
        self.dense_three_norms = [tf.keras.layers.BatchNormalization(
            ) for i in range(num_layers)]
        self.output_layer = tf.keras.layers.Dense(output_size, use_bias=False)
    
    def __call__(self, sequence_one, sequence_two):
        depth_ids = tf.cast(tf.range(self.hidden_size, delta=2), tf.float32)
        sequence_one_ids = tf.cast(tf.range(tf.shape(sequence_one)[1]), tf.float32)
        sequence_two_ids = tf.cast(tf.range(tf.shape(sequence_two)[1]), tf.float32)
        depth_one_ids, sequence_one_ids = tf.meshgrid(depth_ids, sequence_one_ids)
        depth_two_ids, sequence_two_ids = tf.meshgrid(depth_ids, sequence_two_ids)
        positional_embeddings_one = tf.concat([
            tf.sin(sequence_one_ids / tf.pow(10000.0, 2.0 * depth_one_ids / self.hidden_size)),
            tf.cos(sequence_one_ids / tf.pow(10000.0, 2.0 * (depth_one_ids + 1) / self.hidden_size))], 1)
        positional_embeddings_two = tf.concat([
            tf.sin(sequence_two_ids / tf.pow(10000.0, 2.0 * depth_two_ids / self.hidden_size)),
            tf.cos(sequence_two_ids / tf.pow(10000.0, 2.0 * (depth_two_ids + 1) / self.hidden_size))], 1)
        sequence_one = self.input_one_layer(sequence_one) + positional_embeddings_one
        sequence_two = self.input_two_layer(sequence_two) + positional_embeddings_two
        for attend_one, attend_norm_one, hidden, output, dense_norm in zip(
                self.attention_one_layers, self.attention_one_norms,
                self.dense_one_hidden_layers, self.dense_one_output_layers,
                self.dense_one_norms):
            sequence_one = attend_norm_one(sequence_one + attend_one(
                sequence_one, sequence_one, sequence_one, use_mask=False))
            sequence_one = dense_norm(sequence_one + output(tf.nn.relu(hidden(sequence_one))))
        for attend_two, attend_norm_two, attend_three, attend_norm_three, hidden, output, dense_norm in zip(
                self.attention_two_layers, self.attention_two_norms,
                self.attention_three_layers, self.attention_three_norms,
                self.dense_one_hidden_layers, self.dense_one_output_layers,
                self.dense_one_norms):
            sequence_two = attend_norm_two(sequence_two + attend_two(
                sequence_two, sequence_two, sequence_two, use_mask=False))
            sequence_two = attend_norm_three(sequence_two + attend_three(
                sequence_two, sequence_one, sequence_one, use_mask=False))
            sequence_two = dense_norm(sequence_two + output(tf.nn.relu(hidden(sequence_two))))
        return self.output_layer(sequence_two)
        
    def trainable_variables(self):
        layer_variables = (self.input_one_layer.trainable_variables + self.input_two_layer.trainable_variables + 
            self.output_layer.trainable_variables)
        for layer in self.attention_one_layers:
            layer_variables += layer.trainable_variables
        for layer in self.attention_one_norms:
            layer_variables += layer.trainable_variables
        for layer in self.dense_one_hidden_layers:
            layer_variables += layer.trainable_variables
        for layer in self.dense_one_output_layers:
            layer_variables += layer.trainable_variables
        for layer in self.dense_one_norms:
            layer_variables += layer.trainable_variables
        for layer in self.attention_two_layers:
            layer_variables += layer.trainable_variables
        for layer in self.attention_two_norms:
            layer_variables += layer.trainable_variables
        for layer in self.dense_two_hidden_layers:
            layer_variables += layer.trainable_variables
        for layer in self.dense_two_output_layers:
            layer_variables += layer.trainable_variables
        for layer in self.dense_two_norms:
            layer_variables += layer.trainable_variables
        for layer in self.attention_three_layers:
            layer_variables += layer.trainable_variables
        for layer in self.attention_three_norms:
            layer_variables += layer.trainable_variables
        for layer in self.dense_three_hidden_layers:
            layer_variables += layer.trainable_variables
        for layer in self.dense_three_output_layers:
            layer_variables += layer.trainable_variables
        for layer in self.dense_three_norms:
            layer_variables += layer.trainable_variables
        return layer_variables
    
    def variables(self):
        layer_variables = (self.input_one_layer.variables + self.input_two_layer.variables + 
            self.output_layer.variables)
        for layer in self.attention_one_layers:
            layer_variables += layer.variables
        for layer in self.attention_one_norms:
            layer_variables += layer.variables
        for layer in self.dense_one_hidden_layers:
            layer_variables += layer.variables
        for layer in self.dense_one_output_layers:
            layer_variables += layer.variables
        for layer in self.dense_one_norms:
            layer_variables += layer.variables
        for layer in self.attention_two_layers:
            layer_variables += layer.variables
        for layer in self.attention_two_norms:
            layer_variables += layer.variables
        for layer in self.dense_two_hidden_layers:
            layer_variables += layer.variables
        for layer in self.dense_two_output_layers:
            layer_variables += layer.variables
        for layer in self.dense_two_norms:
            layer_variables += layer.variables
        for layer in self.attention_three_layers:
            layer_variables += layer.variables
        for layer in self.attention_three_norms:
            layer_variables += layer.variables
        for layer in self.dense_three_hidden_layers:
            layer_variables += layer.variables
        for layer in self.dense_three_output_layers:
            layer_variables += layer.variables
        for layer in self.dense_three_norms:
            layer_variables += layer.variables
        return layer_variables
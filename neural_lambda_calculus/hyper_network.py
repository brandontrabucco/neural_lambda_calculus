"""Author: Brandon Trabucco, Kavi Gupta, Copyright 2019"""


import neural_lambda_calculus.module as module
import tensorflow as tf
import numpy as np


class HyperNetwork(module.Module):

    def __init__(self, hidden_size, num_layers, output_shapes):
        self.hidden_layers = [tf.keras.layers.Dense(hidden_size) for i in range(num_layers)]
        self.norm_layers = [tf.keras.layers.BatchNormalization() for i in range(num_layers)]
        self.output_layers = [tf.keras.layers.Dense(int(np.prod(shape))) for shape in output_shapes]
        self.output_shapes = output_shapes
    
    def __call__(self, x):
        for i, (layer, norm) in enumerate(list(zip(self.hidden_layers, self.norm_layers))):
            x = norm(tf.nn.relu(layer(x))) + (0 if i == 0 else x)
        return [tf.reshape(layer(x), [tf.shape(x)[0]] + list(shape)) for layer, shape in zip(
            self.output_layers, self.output_shapes)]
        
    def trainable_variables(self):
        module_variables = []
        for layer in self.hidden_layers:
            module_variables += layer.trainable_variables
        for layer in self.norm_layers:
            module_variables += layer.trainable_variables
        for layer in self.output_layers:
            module_variables += layer.trainable_variables
        return module_variables
    
    def variables(self):
        module_variables = []
        for layer in self.hidden_layers:
            module_variables += layer.variables
        for layer in self.norm_layers:
            module_variables += layer.variables
        for layer in self.output_layers:
            module_variables += layer.variables
        return module_variables
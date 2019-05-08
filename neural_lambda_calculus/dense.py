"""Author: Brandon Trabucco, Kavi Gupta, Copyright 2019"""


import tensorflow as tf


def dense(features, weights, biases=None, activation=(lambda x: x)):
    while len(weights.shape) < len(features.shape):
        weights = tf.expand_dims(weights, -3)
    if biases is not None:
        while len(biases.shape) < len(features.shape):
            biases = tf.expand_dims(biases, -2)
    weights = tf.broadcast_to(weights, [
        tf.shape(features)[i] for i in range(len(features.shape) - 2)] + [tf.shape(weights)[-2], tf.shape(weights)[-1]])
    return activation(tf.matmul(features, weights) + (0 if biases is None else biases))
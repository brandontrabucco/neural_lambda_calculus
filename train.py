"""Author: Brandon Trabucco, Kavi Gupta, Copyright 2019"""


import neural_lambda_calculus.runtime as runtime
import tensorflow as tf


if __name__ == "__main__":


    seed = tf.random_normal([32, 128])
    environment = tf.random_normal([32, 4, 128])

    runtime = runtime.Runtime(10, 128, 3, 
        3 * [(128, 128), (128, 128), (128, 128), (128, 128), 
            (128, 128), (1, 128), (128, 128), (1, 128)] + [(128, 257), (1, 257)], 
        [4, 4, 4], 
        [32, 32, 32])

    x = runtime(seed, environment)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run(x))
"""Author: Brandon Trabucco, Kavi Gupta, Copyright 2019"""


import neural_lambda_calculus.runtime as runtime
import tensorflow as tf
import time


if __name__ == "__main__":

    batch_size = 32

    seed = tf.random_normal([batch_size, 128])
    environment = tf.random_normal([batch_size, 4, 128])

    runtime = runtime.Runtime(3, 3, 4, 32, 256, 3, 128, 257)
    x = runtime(seed, environment)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    start_time = time.time()
    for i in range(100):
        sess.run(x)
        end_time = time.time()
        print("{:.5f} [example/sec]".format(batch_size / (end_time - start_time)))
        start_time = end_time
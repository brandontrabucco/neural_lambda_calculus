"""Author: Brandon Trabucco, Kavi Gupta, Copyright 2019"""


class Module(object):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def trainable_variables(self):
        raise NotImplementedError()

    def variables(self):
        raise NotImplementedError()
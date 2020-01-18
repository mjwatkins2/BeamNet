from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import layers

class SpatialPowers(layers.Layer):
    """A layer that can compute integer powers of features (or spatial variables) in a neural network with no nonlinearities. """

    # TODO it was intended to have an input to specify max power of x, but
    # I found and filed several tensorflow bugs, so will have to come back to this later.
    # https://github.com/tensorflow/tensorflow/issues/35011
    # https://github.com/tensorflow/tensorflow/issues/34947
    # https://github.com/tensorflow/tensorflow/issues/35012

    def call(self, inputs):
        x = inputs
        x2 = x*x
        x3 = x2*x
        x4 = x3*x
        return tf.concat([x, x2, x3, x4], axis=1)   # expand features along axis 1, since axis 0 is the batch axis
        
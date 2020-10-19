# ---------------------------
# Alexander Camuto, Matthew Willetts -- 2019
# The University of Oxford, The Alan Turing Institute
# contact: acamuto@turing.ac.uk, mwilletts@turing.ac.uk
# ---------------------------
"""Functions to preprocess mnist data
"""
import numpy as np
import tensorflow as tf

import os
import sys
import shutil
import zipfile
from six.moves import urllib
import scipy.misc
import cv2
import pickle


def data_generator_train(batch_size):
    """
    Generates an infinite sequence of data

    Args:
      x: training data
      y: training labels
      batch_size: batch size to yield

    Yields:
      tuples of x,y pairs each of size batch_size

    """

    while True:
        # --- Randomly select batch_size elements from the training set
        x_batch = np.random.rand(batch_size, 32, 32, 3)
        y_batch = np.random.rand(batch_size, 1)
        # --- Now yield
        yield (x_batch, y_batch)


def data_generator_eval(batch_size):
    """
    Generates an infinite sequence of data

    Args:
      x: training data
      y: training labels
      batch_size: batch size to yield

    Yields:
      tuples of x,y pairs each of size batch_size

    """

    for _ in range(10000):
        # --- Randomly select batch_size elements from the training set
        x_batch = np.random.rand(batch_size, 32, 32, 3)
        y_batch = np.random.rand(batch_size, 1)
        # --- Now yield
        yield (x_batch, y_batch)


def build_input_fns(params):
    """Builds an Iterator switching between train and heldout data."""

    def gen_train():
        return data_generator_train(params["B"])

    def gen_eval():
        return data_generator_eval(params["B"])

    def train_input_fn():
        # Build an iterator over training batches.
        dataset = tf.data.Dataset.from_generator(
            gen_train, (tf.float32, tf.int32),
            (tf.TensorShape([params["B"], 32, 32, 3
                             ]), tf.TensorShape([params["B"], 1])))
        dataset = dataset.prefetch(1)
        return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

    def eval_input_fn():
        # Build an iterator over training batches.
        dataset = tf.data.Dataset.from_generator(
            gen_eval, (tf.float32, tf.int32),
            (tf.TensorShape([params["B"], 32, 32, 3
                             ]), tf.TensorShape([params["B"], 1])))
        dataset = dataset.prefetch(1)
        return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

    # Build an iterator over the heldout set.

    return train_input_fn, eval_input_fn, 10000.0

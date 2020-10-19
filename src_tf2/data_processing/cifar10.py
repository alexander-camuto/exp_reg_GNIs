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


def data_generator_train(x, y, batch_size):
    """
    Generates an infinite sequence of data

    Args:
      x: training data
      y: training labels
      batch_size: batch size to yield

    Yields:
      tuples of x,y pairs each of size batch_size

    """

    num = x.shape[0]
    while True:
        # --- Randomly select batch_size elements from the training set
        idx = np.random.randint(0, num, batch_size)
        x_batch = x[idx]
        y_batch = y[idx]
        # --- Now yield
        yield (x_batch, y_batch)


def data_generator_eval(x, y, batch_size):
    """
    Generates an infinite sequence of test data

    Args:
      x: test data
      y: test labels
      batch_size: batch size to yield

    Yields:
      tuples of x,y pairs each of size batch_size

    """
    num = x.shape[0]
    n_batches = int(num / batch_size)
    for i in range(n_batches):
        idx = list(range(i * batch_size, (i + 1) * batch_size))
        x_batch = x[idx]
        y_batch = y[idx]
        yield (x_batch, y_batch)


def load_cifar10(normalize=True, dequantify=True):
    '''
    :param dataset:
    :param normalize:
    :param dequantify: Add uniform noise to dequantify the data following
        Uria et. al 2013
        "RNADE: The real-valued neural autoregressive density-estimator"
    :return:
    '''
    (train_x, train_y), (test_x,
                         test_y) = tf.keras.datasets.cifar10.load_data()

    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
    train_y = tf.keras.utils.to_categorical(train_y.astype('int32'), 10)
    test_y = tf.keras.utils.to_categorical(test_y.astype('int32'), 10)

    if dequantify:
        train_x += np.random.uniform(0, 1,
                                     size=train_x.shape).astype('float32')
        test_x += np.random.uniform(0, 1, size=test_x.shape).astype('float32')

    if normalize:
        normalizer = train_x.max().astype('float32')
        train_x = train_x / normalizer
        test_x = test_x / normalizer

    return train_x, train_y, test_x, test_y


def build_input_fns(params):
    """Builds an Iterator switching between train and heldout data."""
    x_train, y_train, x_test, y_test = load_cifar10()

    def gen_train():
        return data_generator_train(x_train, y_train, params["B"])

    def gen_eval():
        return data_generator_eval(x_test, y_test, params["B"])

    def train_input_fn():
        # Build an iterator over training batches.
        dataset = tf.data.Dataset.from_generator(
            gen_train, (tf.float32, tf.int32), (tf.TensorShape(
                [params["B"], 32, 32, 3]), tf.TensorShape([params["B"], 10])))
        dataset = dataset.prefetch(1)
        return dataset.make_one_shot_iterator().get_next()

    def eval_input_fn():
        # Build an iterator over training batches.
        dataset = tf.data.Dataset.from_generator(
            gen_eval, (tf.float32, tf.int32), (tf.TensorShape(
                [params["B"], 32, 32, 3]), tf.TensorShape([params["B"], 10])))
        dataset = dataset.prefetch(1)
        return dataset.make_one_shot_iterator().get_next()

    # Build an iterator over the heldout set.

    return train_input_fn, eval_input_fn, x_train.shape[0]

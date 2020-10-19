# ---------------------------
# Alexander Camuto, Matthew Willetts -- 2019
# The University of Oxford, The Alan Turing Institute
# contact: acamuto@turing.ac.uk, mwilletts@turing.ac.uk
# ---------------------------
"""Functions to preprocess mnist data
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle


def drop_dimensions(x_train, x_test, threshold=0.1):
    """
    Removes dimensions with low variance

    Args:
      x_train: training data
      x_test: test data
      threshold: variance threshold for removing dimensions

    Returns:
      x_train: filtered training data
      x_test: filtered test data
      good_dims: dimensions that were retained, by index

    """
    stds = np.std(x_train, axis=0)
    good_dims = np.where(stds > threshold)[0]
    x_train = x_train[:, good_dims]
    x_test = x_test[:, good_dims]
    return x_train, x_test, good_dims


def load_line_reg():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape([-1, 28, 28, 1]) / 256.
    x_test = x_test.reshape([-1, 28, 28, 1]) / 256.
    y_train, y_test = tf.keras.utils.to_categorical(
        y_train, 10), tf.keras.utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


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


def build_input_fns(params):
    """Builds an Iterator switching between train and heldout data."""

    with open('dataset.pkl', 'rb') as f:
        data = pickle.load(f)

    x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data[
        'x_test'], ['y_test']

    def gen_train():
        return data_generator_train(x_train, y_train, params["B"])

    def gen_eval():
        return data_generator_eval(x_test, y_test, params["B"])

    def train_input_fn():
        # Build an iterator over training batches.
        dataset = tf.data.Dataset.from_generator(
            gen_train, (tf.float64, tf.int64),
            (tf.TensorShape([params["B"], 199, 1081, 1
                             ]), tf.TensorShape([params["B"], 1])))
        dataset = dataset.prefetch(1)
        return dataset.make_one_shot_iterator().get_next()

    def eval_input_fn():
        # Build an iterator over training batches.
        dataset = tf.data.Dataset.from_generator(
            gen_eval, (tf.float32, tf.int32),
            (tf.TensorShape([params["B"], 199, 1081, 1
                             ]), tf.TensorShape([params["B"], 1])))
        dataset = dataset.prefetch(1)
        return dataset.make_one_shot_iterator().get_next()

    # Build an iterator over the heldout set.

    return train_input_fn, eval_input_fn, x_train.shape[0]

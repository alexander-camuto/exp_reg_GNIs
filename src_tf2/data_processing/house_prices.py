# ---------------------------
# Alexander Camuto, Matthew Willetts -- 2019
# The University of Oxford, The Alan Turing Institute
# contact: acamuto@turing.ac.uk, mwilletts@turing.ac.uk
# ---------------------------
"""Functions to preprocess house price data
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

from sklearn import preprocessing


def load_house_prices():

    (x_train,
     y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()

    scaler = preprocessing.StandardScaler().fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    x_train, x_test = np.reshape(x_train, (-1, 13, 1, 1)), np.reshape(
        x_test, (-1, 13, 1, 1))

    y_train, y_test = 2 * (np.reshape(x_train, (-1, 1)) / 50. -
                           0.5), 2 * (np.reshape(x_test, (-1, 1)) / 50. - 0.5)

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
    if batch_size > num:
        batch_size = num
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
        idx = np.random.choice(list(range(num)), batch_size, replace=False)
        x_batch = x[idx]
        y_batch = y[idx]
        # --- Now yield
        yield (x_batch, y_batch)


def build_input_fns(params):
    """Builds an Iterator switching between train and heldout data."""

    x_train, y_train, x_test, y_test = load_house_prices()

    def gen_train():
        return data_generator_train(x_train, y_train, params["B"])

    def gen_eval():
        return data_generator_eval(x_test, y_test, params["B"])

    def train_input_fn():
        # Build an iterator over training batches.
        dataset = tf.data.Dataset.from_generator(
            gen_train, (tf.float64, tf.int64), (tf.TensorShape(
                [params["B"], 13, 1, 1]), tf.TensorShape([params["B"], 1])))
        dataset = dataset.prefetch(1)
        return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

    def eval_input_fn():
        # Build an iterator over training batches.
        eval_size = min(params["B"], x_test.shape[0])
        dataset = tf.data.Dataset.from_generator(
            gen_eval, (tf.float32, tf.int32), (tf.TensorShape(
                [eval_size, 13, 1, 1]), tf.TensorShape([eval_size, 1])))
        dataset = dataset.prefetch(1)
        return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

    # Build an iterator over the heldout set.

    return train_input_fn, eval_input_fn, x_train.shape[0]

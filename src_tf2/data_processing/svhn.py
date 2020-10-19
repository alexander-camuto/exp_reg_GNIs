# ---------------------------
# Alexander Camuto, Matthew Willetts -- 2019
# The University of Oxford, The Alan Turing Institute
# contact: acamuto@turing.ac.uk, mwilletts@turing.ac.uk
# ---------------------------
"""Functions to preprocess SVHN data
"""
import numpy as np
import tensorflow as tf

import os
import sys
import shutil
import zipfile
import scipy.misc
import scipy.io as sio
import pickle as Pkl
import gzip, tarfile
import re, string, fnmatch
import urllib.request


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
        # idx = np.random.randint(0, num, batch_size)
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


def build_input_fns(params, extra=False):
    """Builds an Iterator switching between train and heldout data."""
    x_train, y_train, x_test, y_test = load_svhn(dataset=params["data_dir"],
                                                 extra=extra)

    #
    # x_train, y_train = x_train[:params["B"]], y_train[:params[
    #     "batch_size"]]

    def gen_train():
        return data_generator_train(x_train, y_train, params["B"])

    def gen_eval():
        return data_generator_eval(x_test, y_test, params["B"])

    def train_input_fn():
        # Build an iterator over training batches.
        dataset = tf.data.Dataset.from_generator(
            gen_train, (tf.float32, tf.int32),
            (tf.TensorShape([params["B"], 32, 32, 3
                             ]), tf.TensorShape([params["B"], 10])))
        dataset = dataset.prefetch(1)
        return dataset.make_one_shot_iterator().get_next()

    def eval_input_fn():
        # Build an iterator over training batches.
        dataset = tf.data.Dataset.from_generator(
            gen_eval, (tf.float32, tf.int32),
            (tf.TensorShape([params["B"], 32, 32, 3
                             ]), tf.TensorShape([params["B"], 10])))
        dataset = dataset.prefetch(1)
        return dataset.make_one_shot_iterator().get_next()

    # Build an iterator over the heldout set.

    return train_input_fn, eval_input_fn, x_train.shape[0]


def _get_datafolder_path():
    full_path = os.path.abspath('.')
    path = full_path + '/data'
    return path


def _unpickle(f):
    import cPickle
    fo = open(f, 'rb')
    d = Pkl.load(fo)
    fo.close()
    return d


def load_svhn(dataset=_get_datafolder_path() + '/svhn/',
              normalize=True,
              dequantify=True,
              extra=False):
    '''
    :param dataset:
    :param normalize:
    :param dequantify: Add uniform noise to dequantify the data following
        Uria et. al 2013
        "RNADE: The real-valued neural autoregressive density-estimator"
    :param extra: include extra svhn samples
    :return:
    '''

    if not os.path.isfile(dataset + 'svhn_train.pkl'):
        datasetfolder = os.path.dirname(dataset + 'svhn_train.pkl')
        if not os.path.exists(datasetfolder):
            os.makedirs(datasetfolder)
        _download_svhn(dataset, extra=False)

    with open(dataset + 'svhn_train.pkl', 'rb') as f:
        train_x, train_y = Pkl.load(f)
    with open(dataset + 'svhn_test.pkl', 'rb') as f:
        test_x, test_y = Pkl.load(f)

    if extra:
        if not os.path.isfile(dataset + 'svhn_extra.pkl'):
            datasetfolder = os.path.dirname(dataset + 'svhn_train.pkl')
            if not os.path.exists(datasetfolder):
                os.makedirs(datasetfolder)
            _download_svhn(dataset, extra=True)

        with open(dataset + 'svhn_extra.pkl', 'rb') as f:
            extra_x, extra_y = Pkl.load(f)
        train_x = np.concatenate([train_x, extra_x])
        train_y = np.concatenate([train_y, extra_y])

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


def _download_svhn(dataset, extra):
    """
    Download the SVHN dataset
    """
    from scipy.io import loadmat

    print('Downloading data from http://ufldl.stanford.edu/housenumbers/, ' \
          'this may take a while...')
    if extra:
        print("Downloading extra data...")
        urllib.request.urlretrieve(
            'http://ufldl.stanford.edu/housenumbers/extra_32x32.mat',
            dataset + 'extra_32x32.mat')
        extra = loadmat(dataset + 'extra_32x32.mat')
        extra_x = extra['X'].swapaxes(2, 3).swapaxes(1, 2).swapaxes(0, 1)
        extra_y = extra['y'].reshape((-1)) - 1

        print("Saving extra data")
        with open(dataset + 'svhn_extra.pkl', 'wb') as f:
            Pkl.dump([extra_x, extra_y], f, protocol=Pkl.HIGHEST_PROTOCOL)
        os.remove(dataset + 'extra_32x32.mat')

    else:
        print("Downloading train data...")
        urllib.request.urlretrieve(
            'http://ufldl.stanford.edu/housenumbers/train_32x32.mat',
            dataset + 'train_32x32.mat')
        print("Downloading test data...")
        urllib.request.urlretrieve(
            'http://ufldl.stanford.edu/housenumbers/test_32x32.mat',
            dataset + 'test_32x32.mat')

        train = loadmat(dataset + 'train_32x32.mat')
        train_x = train['X'].swapaxes(2, 3).swapaxes(1, 2).swapaxes(0, 1)
        train_y = train['y'].reshape((-1)) - 1
        test = loadmat(dataset + 'test_32x32.mat')
        test_x = test['X'].swapaxes(2, 3).swapaxes(1, 2).swapaxes(0, 1)
        test_y = test['y'].reshape((-1)) - 1

        print("Saving train data")
        with open(dataset + 'svhn_train.pkl', 'wb') as f:
            Pkl.dump([train_x, train_y], f, protocol=Pkl.HIGHEST_PROTOCOL)
        print("Saving test data")
        with open(dataset + 'svhn_test.pkl', 'wb') as f:
            Pkl.dump([test_x, test_y], f, protocol=Pkl.HIGHEST_PROTOCOL)
        os.remove(dataset + 'train_32x32.mat')
        os.remove(dataset + 'test_32x32.mat')

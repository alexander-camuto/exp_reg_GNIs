# ---------------------------
# Alexander Camuto, Matthew Willetts -- 2019
# The University of Oxford, The Alan Turing Institute
# contact: acamuto@turing.ac.uk, mwilletts@turing.ac.uk
# ---------------------------
"""Functions to preprocess all data
"""
import numpy as np

from .random import build_input_fns as build_input_fns_random
from .mnist import build_input_fns as build_input_fns_mnist
from .fmnist import build_input_fns as build_input_fns_fmnist
from .cifar10 import build_input_fns as build_input_fns_cifar10
from .svhn import build_input_fns as build_input_fns_svhn
from .line_reg import build_input_fns as build_input_fns_line_reg
from .house_prices import build_input_fns as build_input_fns_house_prices


def load_dataset(dataset, params):
    # --- Good ol random data to test the pipeline
    if dataset == 'random':
        image_shape = [32, 32, 3]
        train_input_fn, eval_input_fn, n_d = build_input_fns_random(params)
    elif dataset == 'mnist':
        image_shape = [28, 28, 1]
        # --- The MNIST database of handwritten digits, available from this page,
        # --- has a training set of 60,000 examples, and a test set of 10,000 examples.
        # --- It is a subset of a larger set available from NIST.
        # --- The digits have been size-normalized and centered in a 28x28x1 image.
        train_input_fn, eval_input_fn, n_d = build_input_fns_mnist(params)
        # --- We filter the image for directions of low variance
    elif dataset == 'fmnist':
        image_shape = [28, 28, 1]
        # --- The MNIST database of handwritten digits, available from this page,
        # --- has a training set of 60,000 examples, and a test set of 10,000 examples.
        # --- It is a subset of a larger set available from NIST.
        # --- The digits have been size-normalized and centered in a 28x28x1 image.
        train_input_fn, eval_input_fn, n_d = build_input_fns_fmnist(params)
        # --- We filter the image for directions of low variance
    elif dataset == 'cifar10':
        # --- 3D faces is a standard ML dataset composed of 50 people where for
        # --- each there are 21 steps over azimuth and 11 over each of elevation
        # --- and lighting. Images are 64x64 greyscale.
        image_shape = [32, 32, 3]
        train_input_fn, eval_input_fn, n_d = build_input_fns_cifar10(params)
    elif dataset == 'svhn':
        # --- 3D faces is a standard ML dataset composed of 50 people where for
        # --- each there are 21 steps over azimuth and 11 over each of elevation
        # --- and lighting. Images are 64x64 greyscale.
        image_shape = [32, 32, 3]
        train_input_fn, eval_input_fn, n_d = build_input_fns_svhn(params)
    elif dataset == 'line_reg':
        # --- 3D faces is a standard ML dataset composed of 50 people where for
        # --- each there are 21 steps over azimuth and 11 over each of elevation
        # --- and lighting. Images are 64x64 greyscale.
        image_shape = [199, 1081, 1]
        train_input_fn, eval_input_fn, n_d = build_input_fns_line_reg(params)
    elif dataset == 'house_prices':
        # --- 3D faces is a standard ML dataset composed of 50 people where for
        # --- each there are 21 steps over azimuth and 11 over each of elevation
        # --- and lighting. Images are 64x64 greyscale.
        image_shape = [13, 1, 1]
        train_input_fn, eval_input_fn, n_d = build_input_fns_house_prices(
            params)
    return train_input_fn, eval_input_fn, image_shape, n_d

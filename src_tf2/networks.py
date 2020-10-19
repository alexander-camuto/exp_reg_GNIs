# ---------------------------
# Alexander Camuto, Matthew Willetts, Umut Şimşekli -- 2019
# The University of Oxford, The Alan Turing Institute
# contact: acamuto@turing.ac.uk, mwilletts@turing.ac.uk, umut.simsekli@stats.ox.ac.uk
# ---------------------------

import tensorflow_probability as tfp
import functools
import itertools

import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Reshape, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Input, Activation, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

from tensorflow.keras import backend as K
from tensorflow.python.ops.parallel_for.gradients import batch_jacobian

tfd = tfp.distributions


def int_shape(x):
    return list(map(int, x.get_shape()))


def square_error(labels, preds):
    return tf.reduce_sum(0.5 * (labels - preds)**2, axis=-1)


def cross_entropy(labels, preds):
    return -tf.reduce_sum(
        labels * tf.math.log(tf.nn.softmax(preds, axis=-1) + 1e-5), axis=-1)


def make_vgg13(activation, input_shape, n_output):

    layer_reg = dict(kernel_regularizer=None, bias_regularizer=None)

    dense = functools.partial(Dense,
                              kernel_initializer='glorot_normal',
                              activation=activation,
                              bias_initializer='zeros',
                              **layer_reg)
    conv = functools.partial(Conv2D,
                             kernel_initializer='glorot_normal',
                             activation=activation,
                             bias_initializer='zeros',
                             padding='SAME',
                             **layer_reg)

    maxpool = functools.partial(MaxPooling2D)

    layers = [
        Input(shape=(np.prod(input_shape), ), name='input'),
        Reshape(input_shape)
    ]

    # Block 1
    layers.append(
        conv(filters=64, kernel_size=3, strides=1, name='block1_conv1'))
    layers.append(
        conv(filters=64, kernel_size=3, strides=1, name='block1_conv2'))
    layers.append(maxpool((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    layers.append(
        conv(filters=128, kernel_size=3, strides=1, name='block2_conv1'))
    layers.append(
        conv(filters=128, kernel_size=3, strides=1, name='block2_conv2'))
    layers.append(maxpool((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    layers.append(
        conv(filters=128, kernel_size=3, strides=1, name='block3_conv1'))
    layers.append(
        conv(filters=128, kernel_size=3, strides=1, name='block3_conv2'))
    layers.append(maxpool((2, 2), strides=(2, 2), name='block3_pool'))

    # # Block 4
    # layers.append(
    #     conv(filters=512, kernel_size=3, strides=1, name='block4_conv1'))
    # layers.append(
    #     conv(filters=512, kernel_size=3, strides=1, name='block4_conv2'))
    # layers.append(maxpool((2, 2), strides=(2, 2), name='block4_pool'))

    layers.append(Flatten())
    # --- Tf losses do softmax internally so keep activation as None
    layers.append(dense(units=256))
    layers.append(dense(units=n_output, activation=None))

    def convnet(x):

        # --- We use the Model and not Sequential API because Sequential makes
        # --- it difficult to access intermediary outputs
        out = [layers[1](layers[0])]
        for l in layers[2:]:
            out.append(l(out[-1]))

        # --- Setup model
        network = Model(inputs=layers[0], outputs=out)
        outputs = network(x)

        activations = [x] + [
            o for o in outputs if any(x in o.name for x in ['dense', 'conv'])
        ]

        layer_shapes = [[np.prod(input_shape)]] + [
            int_shape(o)[1:]
            for o in outputs if any(x in o.name for x in ['dense', 'conv'])
        ]

        # --- outputs contains activations (odd numbered layers) and logits (even numbered layers)
        return dict(net=network,
                    outputs=outputs,
                    activations=activations,
                    layer_shapes=layer_shapes)

    return convnet


def make_convnet(activation, input_shape, n_output=1):

    dense = functools.partial(Dense,
                              kernel_initializer='glorot_normal',
                              activation=activation,
                              bias_initializer='zeros')
    conv = functools.partial(Conv2D,
                             kernel_initializer='glorot_normal',
                             activation=activation,
                             bias_initializer='zeros',
                             padding='SAME')

    layers = [
        Input(shape=(np.prod(input_shape), ), name='input'),
        Reshape(input_shape)
    ]

    layers.append(conv(filters=32, kernel_size=4, strides=2))
    layers.append(conv(filters=128, kernel_size=4, strides=2))
    layers.append(Flatten())
    # --- Tf losses do softmax internally so keep activation as None
    layers.append(dense(units=n_output, activation=None))

    def convnet(x):
        # --- We use the Model and not Sequential API because Sequential makes
        # --- it difficult to access intermediary outputs
        out = [layers[1](layers[0])]
        for l in layers[2:]:
            out.append(l(out[-1]))

        # --- Setup model
        network = Model(inputs=layers[0], outputs=out)
        outputs = network(x)

        activations = [x] + [
            o for o in outputs if any(x in o.name for x in ['dense', 'conv'])
        ]

        layer_shapes = [[np.prod(input_shape)]] + [
            int_shape(o)[1:] for o in outputs if 'conv' in o.name
        ]

        # --- outputs contains activations (odd numbered layers) and logits (even numbered layers)
        return dict(net=network,
                    outputs=outputs,
                    activations=activations,
                    layer_shapes=layer_shapes)

    return convnet


def make_mlp(activation, input_shape, N, H, n_output=1):
    """ Creates the discriminant function.

  Args:
    activation: Activation function in hidden layers.
    input_shape: Dimensionality of the input.
    N: Number of layers
    H: Size of hidden layers (# of neurons)

  Returns:
    mlp: A `callable` mapping a `Tensor` of inputs to a
      prediction over classes.
  """
    dense = functools.partial(Dense,
                              kernel_initializer='glorot_normal',
                              activation=activation,
                              use_bias=False)

    layers = [Input(shape=(input_shape, ), name='input')]

    for i in range(N):
        layers.append(dense(units=H))
    # --- Tf losses do softmax internally so keep activation as None
    layers.append(dense(units=n_output, activation=None, use_bias=False))

    def mlp(x):

        # --- We use the Model and not Sequential API because Sequential makes
        # --- it difficult to access intermediary outputs
        out = [layers[1](layers[0])]
        for l in layers[2:]:
            out.append(l(out[-1]))

        # --- Setup model
        network = Model(inputs=layers[0], outputs=out)
        outputs = network(x)
        if type(outputs) is not list:
            outputs = [outputs]

        # --- if linear
        activations = [x] + [
            o for o in outputs if any(n in o.name for n in ['dense'])
        ]

        layer_shapes = [[np.prod(input_shape)]] + [
            int_shape(o)[1:] for o in outputs if 'dense' in o.name
        ]

        # --- outputs contains activations (odd numbered layers) and logits (even numbered layers)
        return dict(net=network,
                    outputs=outputs,
                    activations=activations,
                    layer_shapes=layer_shapes)

    return mlp


def gen_noise(a, sigma, mode, n_samples, seed=0, p=0.0):
    shape = [n_samples] + int_shape(a)
    if mode == 'mult':
        sigma *= a
    noise = tf.random.normal(shape,
                             mean=0.0,
                             stddev=sigma,
                             dtype=tf.dtypes.float32,
                             name='noise')
    r = tf.random.uniform(shape=shape, maxval=1)
    b = tf.cast(tf.math.greater(p, r), dtype=tf.float32)
    return noise


def replace_mask_layer(x,
                       model,
                       non_targeted_layers=[],
                       var=1.0,
                       n_samples=1,
                       mode='add'):
    """ Adds / replaces dropout layers in the discriminant network

  Args:
    X_data: Features data used as input to discriminant model
    model: Keras Model into which we are inserting new 'mask' layers
    dropout_mask: Dropout mask for each layer of dim (batch_size, input_size + N*H)
    traces_layers: sample 'traces' for a given layer either via VI or MCMC
    input_size: Dimensionality of the input.
    N: Number of layers
    H: Size of hidden layers (# of neurons)

  Returns:
    logits: logits of new Model for input data X_data
  """

    # --- Retrieve the old model's layers
    data = x
    sigma = np.sqrt(var)
    layers = [l for l in model['net'].layers]
    layer_set = list(set(range(len(layers))) - set(non_targeted_layers))

    noise_gen = iter([
        gen_noise(a, sigma, mode, n_samples, seed=i)
        if i in layer_set else tf.zeros([n_samples] + int_shape(a))
        for i, a in enumerate(model['activations'])
    ])

    # --- Sequentially mask each layer's activations and noise the missing values,
    # --- up until the penultimate logit layer (we don't noise the output layer)
    noises, activations, x = [], [], data

    for i, l in enumerate(layers[:-1]):
        x = l(x)
        if any(n in l.name for n in ['input', 'conv', 'dense']):
            noises.append(next(noise_gen))
            if 'input' in l.name:
                x = Lambda(lambda x: x + noises[-1])(x)
                activations.append(x)
                x = tf.reshape(x, [-1] + int_shape(x)[2:])
            else:
                x = Lambda(lambda x: x + tf.reshape(noises[-1], [-1] +
                                                    int_shape(x)[1:]))(x)
                activations.append(
                    tf.reshape(x, [n_samples, -1] + int_shape(x)[1:]))
    noises.append(next(noise_gen))
    pred = layers[-1](x)
    activations.append(tf.reshape(pred, [n_samples, -1] + int_shape(pred)[1:]))

    return dict(activations=activations, noise=noises)


def perturbative_solution(x, model, loss, EC, var, loss_fn):
    """ Adds / replaces dropout layers in the discriminant network

  Args:
    X_data: Features data used as input to discriminant model
    model: Keras Model into which we are inserting new 'mask' layers
    dropout_mask: Dropout mask for each layer of dim (batch_size, input_size + N*H)
    traces_layers: sample 'traces' for a given layer either via VI or MCMC
    input_size: Dimensionality of the input.
    N: Number of layers
    H: Size of hidden layers (# of neurons)

  Returns:
    logits: logits of new Model for input data X_data
  """
    from .pyhessian import HessianEstimator

    # --- Retrieve the old model's layers
    model_copy = tf.keras.models.clone_model(model)

    layers = [l for l in model.layers]

    new_weights = []

    weights = [l.trainable_weights[0] for l in layers[1:]]
    H = HessianEstimator(None, loss, None, weights, None, None, 404).get_H_op()
    H_inv = tf.linalg.inv(H)
    J = tf.concat(
        [tf.reshape(tf.gradients([EC], [w])[0], (-1, 1)) for w in weights],
        axis=0)
    update = H_inv @ J

    print(update)

    start = 0
    for i, w in enumerate(weights):
        num_w = np.prod(int_shape(w))
        DW = tf.reshape(update[start:num_w + start], w.shape)
        DW = tf.Print(DW, [DW], 'DW')
        new_w = w - DW
        start += num_w
        # x = x @ new_w
        if i < len(layers) - 1:
            x = tf.keras.activations.elu(x @ new_w)
        else:
            x = x @ new_w

    return tf.reduce_mean(loss_fn(x))


def heavy_tail_variance(Js, loss, preds):

    # --- Here we estimate each component of the regularizer
    dL_dhL = tf.gradients([loss], [preds])[0]
    H_L = batch_jacobian(dL_dhL, preds, use_pfor=False)

    H_var = 0
    # --- The is the Heavy tailed noise `'Covariance'` variance
    for (J_1, J_2) in itertools.permutations(Js, 2):
        H_var += 0.5 * tf.reduce_sum(J_1 @ tf.transpose(J_2, [0, 2, 1]),
                                     axis=[-2, -1])
    return H_var


def calc_taylor_expansion(Js, loss, preds, noises, B, n_samples):

    noisy_Js = [noises[i] @ J for i, J in enumerate(Js)]

    dL_dhL = tf.gradients([loss], [preds])[0]
    H_L = batch_jacobian(dL_dhL, preds, use_pfor=False)

    G, C, H = 0, 0, 0
    # --- This is the Gaussian noise
    for J in noisy_Js:
        G += tf.reduce_sum(tf.reshape(J, (n_samples, B, -1)) * dL_dhL,
                           axis=[0, -1]) / n_samples

    # --- The is the Chi-Squared `'Covariance'` noise
    for (J1, J2) in zip(noisy_Js, noisy_Js):
        C += 0.5 * tf.reduce_sum(J1 @ H_L @ tf.transpose(J2, [0, 2, 1]),
                                 axis=[-2, -1]) / n_samples

    # --- The is the Heavy tailed noise `'Covariance'` noise
    for (J1, J2) in itertools.permutations(noisy_Js, 2):
        H += 0.5 * tf.reduce_sum(J1 @ H_L @ tf.transpose(J2, [0, 2, 1]),
                                 axis=[-2, -1]) / n_samples

    return G, C, H


def calc_tikhonov_reg(Js, acts, preds, noise_mode, var, loss_type):
    l_noise = 0
    n_output = int_shape(preds)[-1]
    for a, J in zip(acts, Js):
        if loss_type == 'cross_entropy':
            # --- Classification loss, log(p(y|x))
            p = tf.nn.softmax(preds, axis=1)
            H_l = tf.linalg.diag(p) - tf.expand_dims(p, 2) @ tf.expand_dims(
                p, 1)
            if noise_mode == 'mult':
                J = tf.tile(tf.expand_dims(a, 2), [1, 1, n_output]) * J
            # EC = tf.reduce_sum(J * (J @ H_l), axis=[-2, -1])
            if 'diag' in noise_mode:
                print("here")
                EC = tf.reduce_sum(tf.linalg.diag_part(
                    H_l * (tf.transpose(J, [0, 2, 1]) @ J)),
                                   axis=[-1])
                print(EC)
            else:
                EC = tf.reduce_sum(H_l * (tf.transpose(J, [0, 2, 1]) @ J),
                                   axis=[-2, -1])
        elif loss_type == 'mse':
            var_l = 1
            if noise_mode == 'mult':
                var_l *= a**2
            EC = tf.reduce_sum(var_l * tf.reduce_sum(J**2, axis=-1), axis=[-1])
        l_noise += 0.5 * var * EC
    return l_noise

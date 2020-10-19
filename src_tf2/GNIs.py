# ---------------------------
# Alexander Camuto, Matthew Willetts, Umut Şimşekli -- 2019
# The University of Oxford, The Alan Turing Institute
# contact: acamuto@turing.ac.uk, mwilletts@turing.ac.uk, umut.simsekli@stats.ox.ac.uk
# ---------------------------

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from .networks import heavy_tail_variance, calc_tikhonov_reg, calc_taylor_expansion, perturbative_solution, make_mlp, make_convnet, make_vgg13, replace_mask_layer, int_shape, cross_entropy, square_error
from .trace_hessian import trace_hessian

from tensorflow.python.ops import array_ops

from tensorflow.python.ops.parallel_for.gradients import batch_jacobian

from scipy.stats import kurtosis
from scipy.stats import skew

tfd = tfp.distributions

from scipy.stats import kurtosis
from scipy.stats import skew


def tensor_skew(l):
    return skew(l[0])


def tensor_kurtosis(l):
    return kurtosis(l[0])


def pack_images(images, rows, cols):
    """Helper utility to make a field of images."""
    shape = tf.shape(input=images)
    width = shape[-3]
    height = shape[-2]
    depth = shape[-1]
    images = tf.reshape(images, (-1, width, height, depth))
    batch = tf.shape(input=images)[0]
    rows = tf.minimum(rows, batch)
    cols = tf.minimum(batch // rows, cols)
    images = images[:rows * cols]
    images = tf.reshape(images, (rows, cols, width, height, depth))
    images = tf.transpose(a=images, perm=[0, 2, 1, 3, 4])
    images = tf.reshape(images, [1, rows * width, cols * height, depth])
    return images


def image_tile_summary(name, tensor, rows=8, cols=8):
    tf.compat.v1.summary.image(name,
                               pack_images(tensor, rows, cols),
                               max_outputs=1)


def GNIs(features, labels, mode, params, config):
    """Builds the model function for use in an estimator.

  Arguments:
    features: The input features for the estimator.
    labels: The labels, unused here.
    mode: Signifies whether it is train or test or predict.
    params: Some hyperparameters as a dictionary.
    config: The RunConfig, unused here.

  Returns:
    EstimatorSpec: A tf.estimator.EstimatorSpec instance.
  """
    del config
    N, H = params["N"], params["H"]
    n_samples = params["n_samples"]

    params["non_targeted_layers"] = []

    if params["input_inject"]:
        params["non_targeted_layers"] = list(range(1, N + 1))

    params["non_targeted_layers"] += [N + 1]

    image_tile_summary("input", features, rows=1, cols=16)

    # --- Ensure input data is flat
    features = tf.reshape(features, (-1, np.prod(params['image_shape'])))
    features = tf.cast(features, dtype=tf.float32)
    if labels is not None:
        labels = tf.cast(labels, dtype=tf.float32)
    else:
        labels = tf.ones_like(features[:, :10], dtype=None)
    B = int_shape(labels)[0]
    n_output = int_shape(labels)[-1]

    if params['activation'] != 'linear':
        activation = getattr(tf.nn, params['activation'])
    else:
        activation = None

    # --- Make discriminator
    if params["disc_type"] == 'mlp':
        mlp = make_mlp(activation, np.prod(params['image_shape']), N, H,
                       n_output)
    if params["disc_type"] == 'convnet':
        mlp = make_convnet(activation, params['image_shape'], n_output)
    if params["disc_type"] == 'vgg':
        mlp = make_vgg13(activation, params['image_shape'], n_output)

    # --- Retrieve intermediate activations, and layer output
    # --- we don't want to mask the final layer so activations doesn't include the output layer
    p_phi_y = mlp(features)

    sel_layer_shapes = [p_phi_y['layer_shapes'][i] for i in range(N + 1)]

    # --- Get Predictions using log(p(y|x))
    preds = p_phi_y['activations'][-1]

    # --- Classification loss, log(p(y|x))
    if params["loss"] == 'cross_entropy':
        loss = cross_entropy(labels, preds)
        pred_class = tf.argmax(input=preds, axis=-1)
        true_class = tf.argmax(input=labels, axis=-1)
        acc = tf.cast(tf.equal(pred_class, true_class), tf.float32)
        tf.compat.v1.summary.scalar("accuracy", tf.reduce_mean(acc))
    elif params["loss"] == 'mse':
        loss = square_error(labels, preds)

    global_step = tf.compat.v1.train.get_or_create_global_step()

    p_phi_y_noisy = replace_mask_layer(
        features,
        p_phi_y,
        non_targeted_layers=params['non_targeted_layers'],
        var=params["var"],
        n_samples=n_samples,
        mode=params["noise_mode"])

    preds_noisy = p_phi_y_noisy['activations'][-1]

    # --- Classification loss, log(p(y|x))
    if params["loss"] == 'cross_entropy':
        noisy_loss = cross_entropy(labels, preds_noisy)
    elif params["loss"] == 'mse':
        noisy_loss = square_error(labels, preds_noisy)

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(
        params["learning_rate"])

    gradients, variables = [], []

    tf.compat.v1.summary.scalar("learning_rate", params["learning_rate"])
    tf.compat.v1.summary.scalar("batch_size", B)

    # --- Enumerate over activation layers, zip automatically removes final
    # --- logit layer

    layers = [
        l for l in p_phi_y['net'].layers
        if ('dense' in l.name or 'conv' in l.name)
    ]

    noises = [
        tf.reshape(n, (B, n_samples, -1)) for n in p_phi_y_noisy['noise'][:-1]
    ]

    weights = [layers[i].trainable_weights[0] for i in range(N + 1)]
    acts = p_phi_y['activations'][:-1]

    Js = [
        tf.reshape(batch_jacobian(preds, a, use_pfor=True), (B, -1, n_output))
        for a in acts
    ]
    print(Js)

    G, C, H = calc_taylor_expansion(Js, loss, preds, noises, B, n_samples)

    EC = calc_tikhonov_reg(Js, acts, preds, params["noise_mode"],
                           params["var"], params["loss"])

    H_sig = heavy_tail_variance(Js, loss, preds)

    l_noise = 0
    if params["noise_type"] is None:
        noisy_loss_estimate = loss
    elif params["noise_type"] == 'input':
        noisy_loss_estimate = noisy_loss
    elif 'full' in params["noise_type"]:
        # --- This is the Gaussian stuff
        assert n_samples == 1
        l_noise += H + G + C
        noisy_loss_estimate = loss + l_noise

    elif 'marginal' in params["noise_type"]:
        # --- Don't ever noise final layer
        assert n_samples == 1
        l_noise = EC
        if 'H' in params["noise_type"]:
            l_noise += H

        if 'C' in params["noise_type"]:
            # alpha, beta, sigma, mu = tf.py_func(
            #     estimate_all_params,
            #     inp=[(C - EC)],
            #     Tout=[tf.float32, tf.float32, tf.float32, tf.float32])
            #
            # tf.compat.v1.summary.scalar('C/alpha', alpha)
            # tf.compat.v1.summary.scalar('C/beta', beta)
            # tf.compat.v1.summary.scalar('C/sigma', sigma)
            # tf.compat.v1.summary.scalar('C/mu', mu)
            # tf.compat.v1.summary.scalar('C', tf.reduce_mean(C - EC))
            # tf.compat.v1.summary.histogram('C', C)
            l_noise += (C - EC)
        if 'G' in params["noise_type"]:
            l_noise += G
        noisy_loss_estimate = loss + l_noise

    actual_noise = tf.reduce_mean(noisy_loss - loss)
    estimated_noise = tf.reduce_mean(noisy_loss_estimate - loss)

    tf.compat.v1.summary.scalar('loss/actual_noise', actual_noise)
    tf.compat.v1.summary.scalar('loss/estimated_noise', estimated_noise)

    tf.compat.v1.summary.scalar("loss/noisy_" + params["loss"],
                                tf.reduce_mean(noisy_loss))
    tf.compat.v1.summary.scalar("loss/og_" + params["loss"],
                                tf.reduce_mean(loss))

    noise_err = tf.reduce_mean(estimated_noise - actual_noise)

    tf.compat.v1.summary.scalar(
        'loss/noise_est_pe',
        tf.abs(noise_err / tf.reduce_mean(actual_noise + 1e-8)))

    tf.compat.v1.summary.scalar('loss/noise_est_mse',
                                tf.abs(tf.reduce_mean(noise_err**2)))

    loss_err = tf.reduce_mean(noisy_loss_estimate - noisy_loss)

    tf.compat.v1.summary.scalar(
        'loss/loss_est_pe',
        tf.abs(loss_err / tf.reduce_mean(noisy_loss + 1e-8)))

    tf.compat.v1.summary.scalar('loss/loss_est_mse',
                                tf.abs(tf.reduce_mean(loss_err**2)))

    if params["L2"] > 0:
        vars = tf.trainable_variables()
        l2_reg = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * params["L2"]
        noisy_loss_estimate += l2_reg
        tf.compat.v1.summary.scalar("loss/L2_reg", l2_reg)
        loss_err = tf.reduce_mean(noisy_loss_estimate - noisy_loss)

    # tf.compat.v1.summary.image('activations_covariance', activation_covariance)
    # g_noise =
    for i, w in enumerate(weights):
        layer_name = "layer_" + str(i)
        num_params = np.prod(int_shape(w))

        a = p_phi_y['activations'][i]
        noisy_a = p_phi_y_noisy['activations'][i]
        inj_noise = noisy_a - a
        print(noisy_a, a)

        # --- Display in tensorboard -- Injected noise stats
        tf.compat.v1.summary.histogram(layer_name + '/injected_noise',
                                       inj_noise)

        n_neurons = int_shape(a)[1]

        tf.compat.v1.summary.histogram(layer_name + '/w', w)
        corr = tfp.stats.correlation(a)
        tf.compat.v1.summary.scalar(layer_name + '/corr', tf.reduce_mean(corr))

        sparsity = tf.reduce_sum(tf.cast(a <= 1e-6, tf.float32))

        # tf.compat.v1.summary.scalar(layer_name + '/lifetime_sparsity',
        #                             sparsity / B)
        tf.compat.v1.summary.scalar(layer_name + '/population_sparsity',
                                    sparsity / (B * n_neurons))

        # --- Retrieve the noise of the gradient of each layer
        # --- = noisy gradients - gradients, this corresponds to
        # --- n_t * gradients where n_t is our noise matrix
        # --- W gradients

        og_W_n = tf.gradients([tf.reduce_mean(noisy_loss)], [w])[0]

        g_W_n = tf.gradients([tf.reduce_mean(noisy_loss_estimate)], [w])[0]
        g = tf.gradients(tf.reduce_mean(loss), w)[0]

        err = -g_W_n + og_W_n
        g_noise = g_W_n - g

        tf.compat.v1.summary.scalar(layer_name + '/mean_grad_noise',
                                    tf.reduce_mean(g_noise))
        tf.compat.v1.summary.histogram(layer_name + '/grad_noise', g_noise)

        tf.compat.v1.summary.scalar(layer_name + '/weights_l2/',
                                    tf.reduce_mean(tf.norm(w)))

        tf.compat.v1.summary.scalar(layer_name + '/grad_est_mse',
                                    tf.reduce_mean((og_W_n - g_W_n)**2))
        tf.compat.v1.summary.scalar(layer_name + '/grad_est_pe',
                                    tf.reduce_mean((-og_W_n + g_W_n) / og_W_n))

        gradients.extend([g_W_n])
        variables.extend([w])

    if i > 0 and params['calc_hessian']:
        # --- Number of parameters does not include batch_size

        hessians = trace_hessian([noisy_loss], weights)
        h_trace = tf.reduce_sum(tf.concat(hessians, axis=1)) / (B * n_samples)

        for i, h in enumerate(hessians):
            layer_name = "layer_" + str(i)
            tf.compat.v1.summary.scalar(layer_name + '/H_trace',
                                        tf.reduce_sum(h) / (B * n_samples))

        tf.compat.v1.summary.scalar('network/H_trace', h_trace)

    # --- Sum all them losses

    loss = tf.reduce_mean(loss)
    noisy_loss = tf.reduce_mean(noisy_loss)

    train_step = optimizer.apply_gradients(zip(gradients, variables),
                                           global_step=global_step)

    if mode == tf.estimator.ModeKeys.PREDICT:
        eval_metrics = {}
        predictions = {
            'preds': tf.nn.softmax(p_phi_y['activations'][-1], axis=1)
        }
        predictions['GCH'] = G + C + H - EC

        for i, J in enumerate(Js):
            predictions['J' + str(i)] = J

            # for i, w in enumerate(weights):
            #     predictions['dGCH' + str(i)] = tf.gradients(
            #         [predictions['GCH']], [w])[0]
        if params['calc_hessian']:
            # --- Number of parameters does not include batch_size

            hessians = trace_hessian([noisy_loss], weights[1:3])
            h_trace = tf.reduce_sum(tf.concat(hessians,
                                              axis=1)) / (B * n_samples)

            predictions['h_trace'] = h_trace

    else:
        predictions = {}
        eval_metrics = {
            "loss/og": tf.compat.v1.metrics.mean(loss),
        }
        if params["loss"] == 'cross_entropy':
            eval_metrics["accuracy"] = tf.compat.v1.metrics.mean(acc)

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      predictions=predictions,
                                      train_op=train_step,
                                      eval_metric_ops=eval_metrics)

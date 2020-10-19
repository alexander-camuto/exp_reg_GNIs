# ---------------------------
# Alexander Camuto, Matthew Willetts, Umut Şimşekli -- 2019
# The University of Oxford, The Alan Turing Institute
# contact: acamuto@turing.ac.uk, mwilletts@turing.ac.uk, umut.simsekli@stats.ox.ac.uk
# ---------------------------

import tensorflow_probability as tfp
import functools

import tensorflow as tf
import numpy as np
import levy

from keras.layers import Reshape, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Input, Activation, Dropout, BatchNormalization
from keras.models import Model
from keras import regularizers

tfd = tfp.distributions

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
# from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops


def trace_hessian(ys,
                  xs,
                  name="hessians",
                  colocate_gradients_with_ops=False,
                  gate_gradients=False,
                  aggregation_method=None,
                  B=1.):
    """Constructs the Hessian of sum of `ys` with respect to `x` in `xs`.
  `hessians()` adds ops to the graph to output the Hessian matrix of `ys`
  with respect to `xs`.  It returns a list of `Tensor` of length `len(xs)`
  where each tensor is the Hessian of `sum(ys)`.
  The Hessian is a matrix of second-order partial derivatives of a scalar
  tensor (see https://en.wikipedia.org/wiki/Hessian_matrix for more details).
  Args:
    ys: A `Tensor` or list of tensors to be differentiated.
    xs: A `Tensor` or list of tensors to be used for differentiation.
    name: Optional name to use for grouping all the gradient ops together.
      defaults to 'hessians'.
    colocate_gradients_with_ops: See `gradients()` documentation for details.
    gate_gradients: See `gradients()` documentation for details.
    aggregation_method: See `gradients()` documentation for details.
  Returns:
    A list of Hessian matrices of `sum(ys)` for each `x` in `xs`.
  Raises:
    LookupError: if one of the operations between `xs` and `ys` does not
      have a registered gradient function.
  """
    # xs = gradients_util._AsList(xs)  # pylint: disable=protected-access
    kwargs = {
        "colocate_gradients_with_ops": colocate_gradients_with_ops,
        "gate_gradients": gate_gradients,
        "aggregation_method": aggregation_method
    }
    # Compute first-order derivatives and iterate for each x in xs.
    hessians = []
    _gradients = tf.gradients(ys, xs, **kwargs)
    for gradient, x in zip(_gradients, xs):
        # change shape to one-dimension without graph branching
        gradient = array_ops.reshape(gradient, [-1])

        # Declare an iterator and tensor array loop variables for the gradients.
        n = array_ops.size(x)
        loop_vars = [
            array_ops.constant(0, tf.dtypes.int32),
            tensor_array_ops.TensorArray(x.dtype, n)
        ]
        # Iterate over all elements of the gradient and compute second order
        # derivatives.
        _, hessian = control_flow_ops.while_loop(
            lambda j, _: j < n, lambda j, result: (
                j + 1,
                result.write(
                    j,
                    tf.reshape(tf.gradients(gradient[j], x)[0], (-1, ))[j])),
            loop_vars)

        _shape = array_ops.shape(x)
        _reshaped_hessian = array_ops.reshape(hessian.stack(),
                                              (1, n)) / tf.cast(n, tf.float32)
        hessians.append(_reshaped_hessian)
    return hessians


def int_shape(x):
    return list(map(int, x.get_shape()))

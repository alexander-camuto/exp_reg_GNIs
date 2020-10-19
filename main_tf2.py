# ---------------------------
# Alexander Camuto, Matthew Willetts, Umut Şimşekli -- 2019
# The University of Oxford, The Alan Turing Institute
# contact: acamuto@turing.ac.uk, mwilletts@turing.ac.uk, umut.simsekli@stats.ox.ac.uk
# ---------------------------

from src_tf2.data_processing.utils import load_dataset

from src_tf2.GNIs import GNIs

import tensorflow as tf
from absl import flags
import os
import pickle
from tensorflow.python import debug as tf_debug
import json
import numpy as np

tf.compat.v1.set_random_seed(0)
np.random.seed(0)

flags.DEFINE_string("loss", default='mse', help="loss, cross_entropy or mse")
flags.DEFINE_string(
    "dataset",
    default='house_prices',
    help="Dataset, choice 'mnist', 'fmnist', 'cifar10','svhn'.")
flags.DEFINE_integer("n_epochs",
                     default=100,
                     help="Number of training epochs to run.")
flags.DEFINE_integer("n_samples", default=1, help="Number of samples to draw.")
flags.DEFINE_float("learning_rate",
                   default=0.001,
                   help="Initial learning rate.")
flags.DEFINE_float("L2", default=0.0, help="L2 penalisation on weights.")
flags.DEFINE_integer("N",
                     default=2,
                     help="Number of hidden layers in discriminant network")
flags.DEFINE_multi_integer(
    "non_targeted_layers",
    default=[],
    help="Layers for which we do not add GNIs. Layer 0 refers to data layer.")
flags.DEFINE_integer("H",
                     default=512,
                     help="Size of hidden layers in discriminant network")
flags.DEFINE_bool("dropout",
                  default=True,
                  help="Dropout for hidden layers AND input")
flags.DEFINE_float("var", default=1.0, help="GNI variance")
flags.DEFINE_string("activation",
                    default="linear",
                    help="Activation function for all hidden layers.")
flags.DEFINE_string("noise_type",
                    default=None,
                    help="Noise type for model, input, gradient, None")
flags.DEFINE_string("noise_mode",
                    default='add',
                    help="Noise node for model, add, mult, None")
flags.DEFINE_string("noise_dist", default="Normal", help="Noise dist")
flags.DEFINE_string("run_name", default='run', help="name of run")
flags.DEFINE_integer("B", default=512, help="Batch size.")
flags.DEFINE_string(
    "data_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vae/data"),
    help="Directory where data is stored (if using real data).")
flags.DEFINE_bool(
    "calc_hessian",
    default=False,
    help="If true, calculates the trace of the hessian for each layer.")
flags.DEFINE_bool("input_inject",
                  default=False,
                  help="If true, only injects noise into data layer.")
flags.DEFINE_string(
    "debug",
    default="",
    help="If tensorboard, connects to tensorboard debug. Else CLI")
flags.DEFINE_string("disc_type",
                    default='mlp',
                    help="type of discriminator to use, convnet or mlp")

FLAGS = flags.FLAGS


def set_up_estimator(estimator,
                     params,
                     steps,
                     train_input_fn,
                     eval_input_fn,
                     config,
                     warm_start_dir=None):

    estimator = tf.estimator.Estimator(estimator,
                                       params=params,
                                       config=tf.estimator.RunConfig(**config),
                                       warm_start_from=warm_start_dir)

    # --- We force the graph to finalize because of some memory leaks we had.
    tf.get_default_graph().finalize()

    # --- Setup our train and eval specs.
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=steps,
                                        hooks=[])

    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      steps=None,
                                      start_delay_secs=0,
                                      throttle_secs=1)

    return estimator, train_spec, eval_spec


def main(argv):
    del argv  # unused

    params = FLAGS.flag_values_dict()

    # --- Get rain and eval functions
    train_input_fn, eval_input_fn, params['image_shape'], n_d = load_dataset(
        params['dataset'], params)

    # --- Calculate number of steps per epoch
    params["epoch_steps"] = int(float(n_d) / float(params["B"])) + 1
    # --- Calculate total number of steps
    params["max_steps"] = int(params["n_epochs"]) * params["epoch_steps"]

    if params["disc_type"] == 'mlp':
        warm_start_dir = './init_weights/' + str(params['N']) + '_' + str(
            params['H']) + '_' + params["dataset"]
    elif params["disc_type"] == 'convnet':
        warm_start_dir = './init_weights/conv_' + params["dataset"]
    elif params["disc_type"] == 'vgg':
        warm_start_dir = './init_weights/vgg_' + params["dataset"]

    config = dict(save_checkpoints_steps=10 * params["epoch_steps"],
                  keep_checkpoint_max=1,
                  save_summary_steps=params["epoch_steps"],
                  model_dir=warm_start_dir)

    # --------------------------- Init Model -------------------------------------
    # --- Run for 1 step to dump out the model to the warm_start_dir if it
    # --- doesn't already exist
    print("--------- Initialising model ------------")

    tf.estimator.train_and_evaluate(*set_up_estimator(
        GNIs, params, 1, train_input_fn, eval_input_fn, config))

    config['model_dir'] = './checkpoints/' + params["run_name"]

    if not os.path.exists(config["model_dir"]):
        os.makedirs(config["model_dir"])
    with open(config["model_dir"] + '/params.json', 'w+') as fp:
        json.dump(params, fp)

    estimator, train_spec, eval_spec = set_up_estimator(
        GradientNoiseInjectLR2,
        params,
        params["max_steps"],
        train_input_fn,
        eval_input_fn,
        config,
        warm_start_dir=warm_start_dir)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # --- We force the graph to finalize because of some memory leaks we had.
    tf.compat.v1.get_default_graph().finalize()

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    tf.compat.v1.app.run()

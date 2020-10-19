# Explicit Regularisation of Gaussian Noise Injections

Code for "Explicit Regularisation of Gaussian Noise Injections" (Neurips 2020).

## Set Up on Linux:

1. Download conda at https://docs.conda.io/en/latest/
2. run `sh setup.sh` on a linux system, preferably Ubuntu
3. run `source activate syn_gen` to enter the conda env
4. to kick off a set of experiments run `python main.py`.

## Set Up on MacOS:

1. Download conda at https://docs.conda.io/en/latest/
2. run `conda create -n syn_gen python=3.6`
3. run `source activate syn_gen` to enter the conda env
4. `pip install -r requirements.txt`
5. to kick off a set of experiments run `python main.py`.


## Experiments

The file, `src_tf2/GNIs.py` details the tensorflow estimator implementation of our experiments for GNIs.

See `scripts` for a sample of a set of experiments runs where we calculate the trace of the Hessian (`calc_hessian=True`) on small MLPs for SVHN, CIFAR10 and Boston House Prices.

One can set `disc_type` to `conv` to run larger convolutional models; and to reduce memory load we recommend setting `calc_hessian=False` on these models, and for larger MLPs.

For an interactive notebook showcasing how GNIs affect the Fourier Transform of a neural network, see `playground/GNIs_Fourier_Domain`.

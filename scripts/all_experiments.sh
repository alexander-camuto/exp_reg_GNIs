
# Regression
CUDA_VISIBLE_DEVICES=0 python main_tf2.py --mode=GradientNoiseInjectLR2 --n_epochs=250 --noise_mode=add --calc_hessian=True --dataset=house_prices --B=404 --H=32 --loss=mse --run_name=NoiseInjectInput_house_prices_elu_add_$1 --activation=elu --noise_type=input --var=$1 &
CUDA_VISIBLE_DEVICES=1 python main_tf2.py --mode=GradientNoiseInjectLR2 --n_epochs=250 --noise_mode=add --calc_hessian=True --dataset=house_prices --B=404 --H=32 --loss=mse --run_name=NoiseInjectMarginal_house_prices_elu_add_$1 --activation=elu --noise_type=marginal --var=$1 &
CUDA_VISIBLE_DEVICES=2 python main_tf2.py --mode=GradientNoiseInjectLR2 --n_epochs=250 --noise_mode=add --calc_hessian=True --dataset=house_prices --B=404 --H=32 --loss=mse --run_name=Baseline_house_prices_elu_add_$1 --activation=elu --var=$1 &
CUDA_VISIBLE_DEVICES=3 python main_tf2.py --mode=GradientNoiseInjectLR2 --n_epochs=250 --noise_mode=add --calc_hessian=True --dataset=svhn --B=512 --H=32 --loss=cross_entropy --run_name=NoiseInjectInput_svhn_elu_add_$1 --activation=elu --noise_type=input --var=$1 &
wait
CUDA_VISIBLE_DEVICES=0 python main_tf2.py --mode=GradientNoiseInjectLR2 --n_epochs=250 --noise_mode=add --calc_hessian=True --dataset=svhn --B=512 --H=32 --loss=cross_entropy --run_name=NoiseInjectMarginal_svhn_elu_add_$1 --activation=elu --noise_type=marginal --var=$1 &
CUDA_VISIBLE_DEVICES=1 python main_tf2.py --mode=GradientNoiseInjectLR2 --n_epochs=250 --noise_mode=add --calc_hessian=True --dataset=svhn --B=512 --H=32 --loss=cross_entropy --run_name=Baseline_svhn_elu_add_$1 --activation=elu --var=$1 &
CUDA_VISIBLE_DEVICES=2 python main_tf2.py --mode=GradientNoiseInjectLR2 --n_epochs=250 --noise_mode=add --calc_hessian=True --dataset=cifar10 --B=512 --H=32 --loss=cross_entropy --run_name=NoiseInjectInput_cifar10_elu_add_0.1 --activation=elu --noise_type=input --var=0.1 &
CUDA_VISIBLE_DEVICES=3 python main_tf2.py --mode=GradientNoiseInjectLR2 --n_epochs=250 --noise_mode=add --calc_hessian=True --dataset=cifar10 --B=512 --H=32 --loss=cross_entropy --run_name=NoiseInjectMarginal_cifar10_elu_add_$1 --activation=elu --noise_type=marginal --var=$1 &
wait
CUDA_VISIBLE_DEVICES=0 python main_tf2.py --mode=GradientNoiseInjectLR2 --n_epochs=250 --noise_mode=add --calc_hessian=True --dataset=cifar10 --B=512 --H=32 --loss=cross_entropy --run_name=Baseline_cifar10_elu_add_$1 --activation=elu --var=$1 &

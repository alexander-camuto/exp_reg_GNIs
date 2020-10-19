#!/bin/bash

conda create -n syn_gen python=3.6
source activate syn_gen
pip install -r requirements.txt
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
curl http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb > lib.deb
sudo dpkg -i lib.deb
rm lib.deb
sudo apt-get update
sudo apt-get install libcudnn7=7.4.1.5-1+cuda9.2
sudo apt-get install libcudnn7-dev=7.4.1.5-1+cuda9.2
pip uninstall tensorflow
sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list
sudo apt-get update
sudo apt-get install cuda-10-0
pip install tensorflow-gpu==1.13.1
pip install opencv-python
pip install scipy
pip install keras

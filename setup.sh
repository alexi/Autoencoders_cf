#!/bin/bash
# Upgrade ubuntu
sudo apt-get -y update upgrade dist-upgrade autoremove clean

# Install CUDA
sudo apt-get install -y build-essential
sudo apt-get -y update
sudo apt-get install -y linux-generic
sudo apt-get install -y libprotobuf-dev  # needed for loadcaffe
mkdir libraries
cd libraries

CUDA_REPO_PKG=cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/$CUDA_REPO_PKG && sudo dpkg -i $CUDA_REPO_PKG

# ML_REPO_PKG=nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
# sudo wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/$ML_REPO_PKG && sudo dpkg -i $ML_REPO_PKG

sudo apt-get -y update
sudo apt-get install -y cuda
echo "export PATH=/usr/local/cuda/bin/:\$PATH; export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:\$LD_LIBRARY_PATH; " >>~/.bashrc && source ~/.bashrc
rm -y $CUDA_REPO_PKG

# Reboot the machine and verify that CUDA is running
# reboot
# nvidia-smi 
# lsmod | grep -i nvidia
# ls -alh /dev | grep -i nvidia

# Install CUDNN

wget https://s3-us-west-1.amazonaws.com/tamberhost/cudnn-8.0-linux-x64-v7.tgz
# wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run
# wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7/prod/8.0_20170802/Ubuntu16_04_x64/libcudnn7_7.0.1.13-1+cuda8.0_amd64-deb?autho=1480818585_1a8ff8e74ddf3f6a9f71662f7d51e1ea&file=cudnn-8.0.tgz
tar -xvf cudnn-8.0-linux-x64-v7.tgz 
sudo cp cuda/include/*.h /usr/local/cuda/include
sudo cp cuda/lib64/*.so* /usr/local/cuda/lib64
rm cudnn-8.0-linux-x64-v7.tgz 

# Install Torch -- http://torch.ch/docs/getting-started.html
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch
TORCH_LUA_VERSION=LUA52 ./install.sh
source ~/.bashrc # not needed?

# cd ~/torch; bash install-deps;
# TORCH_LUA_VERSION=LUA52 ./install.sh


# sudo apt install -y luarocks
# Install all reqd torch packages
luarocks install cutorch # CUDA support for torch
 # CUDA neural net?
luarocks install cunn
luarocks install hdf5
luarocks install cudnn
luarocks install loadcaffe

#torch/nn
luarocks install nn #installed with torch, not needed?
luarocks install nnsparse
luarocks install optim

#xlua
torch-rocks install xlua

cd ~/ && git clone https://github.com/dinosaurcop/Autoencoders_cf.git
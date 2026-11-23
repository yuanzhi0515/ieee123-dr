#!/bin/bash

# pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip install \
        virtualenv \
        virtualenvwrapper \
        --no-cache-dir
echo -e "\n# virtualenvwrapper" | tee -a /root/.zshrc /root/.bashrc
echo "export WORKON_HOME=/root/.virtualenvs" | tee -a /root/.zshrc /root/.bashrc
echo "export VIRTUALENVWRAPPER_VIRTUALENV=`which virtualenv`" | tee -a /root/.zshrc /root/.bashrc
echo "export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3.9" | tee -a /root/.zshrc /root/.bashrc
echo "source `which virtualenvwrapper.sh`" | tee -a /root/.zshrc /root/.bashrc
source `which virtualenvwrapper.sh`

mkvirtualenv --system-site-packages ieee123

pip install pandapower==2.14.6
pip install wandb==0.16.6
pip install "stable_baselines3==2.0.0a1"
pip install tyro==0.8.2
pip install matplotlib==3.8.4
pip install tensorboard==2.10
pip install numba==0.56.0
pip install seaborn==0.13.2
pip install numpy==1.22.4
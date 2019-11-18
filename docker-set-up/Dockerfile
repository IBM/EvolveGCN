FROM nvidia/cuda:10.1-base-ubuntu16.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update; 

RUN apt-get install -y wget; \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda;\
    eval "$(/root/miniconda/bin/conda shell.bash hook)";\
    conda init; \
    pip install pyyaml; \
    conda install -y pytorch cudatoolkit=10.0 -c pytorch; \
    conda install -y matplotlib pandas scikit-learn; 

WORKDIR /evolveGCN
ENTRYPOINT /bin/bash


EvolveGCN
=====

This repository contains the code for [EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs](https://arxiv.org/abs/1902.10191), published in AAAI 2020.

## Data

7 datasets were used in the paper:

- stochastic block model: See the 'data' folder. Untar the file for use.
- bitcoin OTC: Downloadable from http://snap.stanford.edu/data/soc-sign-bitcoin-otc.html
- bitcoin Alpha: Downloadable from http://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html
- uc_irvine: Downloadable from http://konect.uni-koblenz.de/networks/opsahl-ucsocial
- autonomous systems: Downloadable from http://snap.stanford.edu/data/as-733.html
- reddit hyperlink network: Downloadable from http://snap.stanford.edu/data/soc-RedditHyperlinks.html
- elliptic: A preprocessed version of https://www.kaggle.com/ellipticco/elliptic-data-set is provided in the following link: ~~https://ibm.box.com/s/j04m8lwoqktjixke2gj7lgllrvvdidme.~~ Untar the file in the 'data' folder for use.

Update on elliptic: The box link is no longer valid. Please see the [instruction](elliptic_construction.md) to manually prepare the preprocessed version.
 
For downloaded data sets please place them in the 'data' folder.

## Requirements
  * PyTorch 1.0 or higher
  * Python 3.6

## Set up with Docker

This docker file describes a container that allows you to run the experiments on any Unix-based machine. GPU availability is recommended to train the models. Otherwise, set the use_cuda flag in parameters.yaml to false.

### Requirements

- [install docker](https://docs.docker.com/install/)
- [install nvidia drivers](https://www.nvidia.com/Download/index.aspx?lang=en-us)

### Installation

#### 1. Build the image

From this folder you can create the image

```sh
sudo docker build -t gcn_env:latest docker-set-up/
```

#### 2. Start the container

Start the container

```sh
sudo docker run -ti  --gpus all -v $(pwd):/evolveGCN  gcn_env:latest
```

This will start a bash session in the container.

## Usage

Set --config_file with a yaml configuration file to run the experiments. For example:

```sh
python run_exp.py --config_file ./experiments/parameters_example.yaml
```

Most of the parameters in the yaml configuration file are self-explanatory. For hyperparameters tuning, it is possible to set a certain parameter to 'None' and then set a min and max value. Then, each run will pick a random value within the boundaries (for example: 'learning_rate', 'learning_rate_min' and 'learning_rate_max').
The 'experiments' folder contains one file for each result reported in the [EvolveGCN paper](https://arxiv.org/abs/1902.10191).

Setting 'use_logfile' to True in the configuration yaml will output a file, in the 'log' directory, containing information about the experiment and validation metrics for the various epochs. The file could be manually analyzed, alternatively 'log_analyzer.py' can be used to automatically parse a log file and to retrieve the evaluation metrics at the best validation epoch. For example:
```sh
python log_analyzer.py log/filename.log
```


## Reference

[1] Aldo Pareja, Giacomo Domeniconi, Jie Chen, Tengfei Ma, Toyotaro Suzumura, Hiroki Kanezashi, Tim Kaler, Tao B. Schardl, and Charles E. Leiserson. [EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs](https://arxiv.org/abs/1902.10191). AAAI 2020.

## BibTeX entry

Please cite the paper if you use this code in your work:


```
@INPROCEEDINGS{egcn,
  AUTHOR = {Aldo Pareja and Giacomo Domeniconi and Jie Chen and Tengfei Ma and Toyotaro Suzumura and Hiroki Kanezashi and Tim Kaler and Tao B. Schardl and Charles E. Leiserson},
  TITLE = {{EvolveGCN}: Evolving Graph Convolutional Networks for Dynamic Graphs},
  BOOKTITLE = {Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence},
  YEAR = {2020},
}
```

# Set Up With Docker

This docker file describes a container that allows you to run the experiments on any Unix-based machine. GPU availability is recommended to train the models. Otherwise, set the use_cuda flag in parameters.yaml to false.

# Requirements

- [install docker](https://docs.docker.com/install/)
- [install nvidia drivers](https://www.nvidia.com/Download/index.aspx?lang=en-us)

# Installation

## 1. Build the image

From this folder you can create the image

```sh
sudo docker build -t gcn_env:latest docker-set-up/
```

## 2. Start the container

Start the container

```sh
sudo docker run -ti  --gpus all -v $(pwd):/evolveGCN  gcn_env:latest
```

This will start a bash session in the container.

## 3. Run an experiment

Run the following command for example:

```sh
python run_exp.py --config_file ./experiments/parameters_uc_irv_mess.yaml
```


# 3d_recognizer

This repository contains a tool to train/test models on 3d point cloud segmentation. It is specifically focussed on recongnizing points on a point cloud eg fingertips.

## Setup
In order to run the tool, the following pre-requisites are required:
* docker (The tool was tested on `Docker version 20.10.17`)
* An Intel Realsense L515 camera (if data capturing is required)
* nvidia drivers should be installed
* nvidia-docker2 should be installed

In order to run the tool, the dockerfile (which is included in the repository) should be build. This can be done by running the `bin/docker_build` script.
This will generate docker image called `3d_gestures`.

By running the `bin/run_in_docker` script, an interactive shell will be opened in a docker container.
As long as this shell is open, the container will keep running. Once the shell is closed, the container is shut down and removed.
## Running locally
In case you are connected to your device directly (screen is attached to the device), you can run your commands in the container shell.
The UI will automatically be forwarded outside the docker container.

## Running remotely
In case you are connect to your device remotely (via ssh), you can connect to the docker container directly in order
to properly forward the UI. This can be done with the command:
```shell
ssh -XC -P 2299 root@<device_ip>
```
The password is set to `abc`. Please note this is not a secure configuration and should not be exposed publicly.

## Usage
_This section assumes a docker container is running and that all commands below are executed inside this docker container shell._

In order to start the 3d_recognizer tool, the following command can be run:
```shell
python main.py
```

Include screenshot

### Commands

Separate from the main UI tool, this repository also includes a few scripts for convenience.
In order to train a new model, one can run `python train.py`. See `python train.py --help` for further information.

It is also possible to evaluate a model by running `python predict.py`. This will run inference a selected model.
See `python predict.py --help` for more information.


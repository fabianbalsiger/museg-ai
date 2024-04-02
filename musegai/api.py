"""API for muscle segmentation."""

from . import train, infer, docker

list_models = docker.list_models
run_model = infer.infer
train_model = train.train

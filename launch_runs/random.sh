#bin#!/bin/sh

CUDA_VISIBLE_DEVICES=3 python train_explainer.py  --wandb True --epochs 50 --name baseline_layer2 --layer layer2
CUDA_VISIBLE_DEVICES=3 python train_explainer.py  --wandb True --epochs 50 --name baseline_layer3 --layer layer3
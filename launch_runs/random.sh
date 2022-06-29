#bin#!/bin/sh

#CUDA_VISIBLE_DEVICES=3 python infer_filter.py  --name baseline_layer4
CUDA_VISIBLE_DEVICES=3 python infer_filter.py  --name baseline_layer4 --layer layer4
CUDA_VISIBLE_DEVICES=3 python infer_filter.py  --name baseline_layer3 --layer layer3
CUDA_VISIBLE_DEVICES=3 python infer_filter.py  --name baseline_layer2 --layer layer2

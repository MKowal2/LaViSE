import torchvision.transforms as tf
import torch
import numpy as np

def unnorm(img, mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])):
    img = img.permute(0,2,3,1)
    img = img * std
    img = img + mean
    return img.permute(0,3,1,2)

def combine_heatmap_img(img, activation, heatmap_opacity=0.60):
    activation = activation/activation.max()
    activation = np.repeat(np.expand_dims(activation, 0), 3, axis=0)
    heatmap_img = heatmap_opacity * activation + (1-heatmap_opacity) * img

    return heatmap_img
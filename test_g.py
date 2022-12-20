import yaml
import os
import sys
import time
import numpy as np
from glob import glob
from math import log
from utils import *
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms 
import torch
from random import shuffle
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models import Discriminator, Generator
from custom import CustomDataset
from matplotlib import pyplot as plt


def generate_random(batch_size, depth):
    random_data = np.random.normal(size=[batch_size, depth]).astype(np.float32)
    random_data = torch.from_numpy(random_data)
    return random_data
    
def read_inference_images(imagelist:list):
    ret = []
    img_size = [28, 28]

    for imgname in imagelist:
        img = Image.open(imgname).convert('L')
        img = img.resize(img_size)
        img = np.array(img).astype(np.float32)
        img = img / 255.0
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        tensor = img.reshape([-1, img_size[0] * img_size[1]])

        ret.append(tensor)

    return ret

if __name__ == "__main__":
    with open("config.yaml", 'r') as config:
        config = yaml.load(config, Loader=yaml.FullLoader)

    _device = torch.device('cpu')

    g = Generator()
    output = g.forward(generate_random(1, 1))
    img = output.detach().numpy().reshape(28, 28)

    plt.imshow(img, interpolation='none', cmap='Blues')
    plt.show()

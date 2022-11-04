import yaml
import os
import sys
import numpy as np
from glob import glob

import torch
from torch import nn
from torch.utils.data import DataLoader

from models import Discriminator, Generator
from custom import CustomDataset


if __name__ == "__main__":
    with open("config.yaml", 'r') as config:
        config = yaml.load(config, Loader=yaml.FullLoader)

    is_cuda = torch.cuda.is_available()
    _device = torch.device('cuda' if is_cuda else 'cpu')

    discriminator = Discriminator()
    generator = Generator()

    d_model = discriminator.model
    g_model = generator.model

    d_model = d_model.to(_device)
    g_model = g_model.to(_device)

    print("===== Discriminator ====")
    print(d_model)
    print("===== Generator ====")
    print(g_model)

    datapath = config['DATASET']['path']
    imgfiles = glob(os.path.join(datapath, "*.png"))
    dataset =  CustomDataset(datapack=imgfiles, img_size=config['DATASET']['img_size'])

    dataloader = DataLoader(dataset=dataset, batch_size=config['TRAIN']['batch_size'])

    criterion = nn.BCELoss()    # Binary Cross Entropy Loss (h(x), y)

    d_optimizer = torch.optim.Adam(d_model.parameters(), lr=float(config['TRAIN']['lr']))
    g_optimizer = torch.optim.Adam(g_model.parameters(), lr=float(config['TRAIN']['lr']))

    for epoch in range(config['TRAIN']['epochs']):
        for step, data in enumerate(dataloader):
            data = data.to(_device)

            rand_data = np.random.rand(config['TRAIN']['batch_size'], 100)
            rand_data = torch.from_numpy(rand_data).to(_device)
            rand_data = rand_data.type(torch.float32)

            d_output = d_model(data)
            g_output = g_model(rand_data)
            print(d_output.shape)
            print(g_output.shape)

            true_target = torch.ones(config['TRAIN']['batch_size'], 1).to(_device)
            fake_target = torch.zeros(config['TRAIN']['batch_size'], 1).to(_device)

            loss = criterion(d_output, true_target) + criterion(d_model(g_output), fake_target)
            loss.backward()
            d_optimizer.step()

            loss = criterion(d_model(g_model(rand_data)), true_target)
            loss.backward()
            g_optimizer.step()

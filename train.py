import yaml
import os
import sys
import time
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

    dataset = CustomDataset(datapack=imgfiles, img_size=config['DATASET']['img_size'])
    dataloader = DataLoader(dataset=dataset, batch_size=config['TRAIN']['batch_size'], drop_last=True)

    criterion = nn.BCELoss()    # Binary Cross Entropy Loss (h(x), y)

    d_optimizer = torch.optim.Adam(d_model.parameters(), lr=float(config['TRAIN']['lr']))
    g_optimizer = torch.optim.Adam(g_model.parameters(), lr=float(config['TRAIN']['lr']))

    total_train_iter = len(dataloader)

    for epoch in range(config['TRAIN']['epochs']):
        epoch_start = time.time()
        train_d_loss, train_g_loss = [], []

        for step, data in enumerate(dataloader):
            iter_start = time.time()
            data = data.reshape([config['TRAIN']['batch_size'], -1])
            data = data.to(_device)

            rand_data = np.random.rand(config['TRAIN']['batch_size'], 100)
            rand_data = torch.from_numpy(rand_data).to(_device)
            rand_data = rand_data.type(torch.float32)

            d_output = d_model(data)
            g_output = g_model(rand_data)
            # print(d_output.shape)
            # print(g_output.shape)

            true_target = torch.ones(config['TRAIN']['batch_size'], 1).to(_device)
            fake_target = torch.zeros(config['TRAIN']['batch_size'], 1).to(_device)

            d_loss = criterion(d_output, true_target) + criterion(d_model(g_output), fake_target)
            d_loss.backward()
            d_optimizer.step()

            g_loss = criterion(d_model(g_model(rand_data)), true_target)
            g_loss.backward()
            g_optimizer.step()

            d_loss = d_loss.detach().cpu().numpy()
            g_loss = g_loss.detach().cpu().numpy()

            train_d_loss.append(d_loss)
            train_g_loss.append(g_loss)

            print("[train %s/%3s] Epoch: %3s | Time: %6.2fs/it | discriminator_loss: %6.4f | generator_loss: %6.4f" % (
                    step + 1, total_train_iter, epoch + 1, time.time() - iter_start, np.round(d_loss, 2), np.round(g_loss, 2)))
        
        train_d_loss = np.round(sum(train_d_loss) / total_train_iter, 2)
        train_g_loss = np.round(sum(train_g_loss) / total_train_iter, 2)

        print("\n[Epoch {} training Ended] > Time: {:.2}s/epoch | Discriminator Loss: {:.4f} | Generator Loss: {:.4f}\n".format(
                            epoch + 1, time.time() - epoch_start, train_d_loss, train_g_loss))
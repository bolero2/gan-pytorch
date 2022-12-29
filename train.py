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
from tqdm import tqdm
import pandas


def generate_random_image(batch_size, depth):
    random_data = np.random.rand(batch_size, depth).astype(np.float32)
    random_data = torch.from_numpy(random_data)
    return random_data

def generate_random_seed(batch_size, depth):
    random_data = np.random.randn(batch_size, depth).astype(np.float32)
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
    epochs = config['TRAIN']['epochs']
    batch_size = config['TRAIN']['batch_size']

    d = Discriminator().to(_device)
    g = Generator().to(_device)

    datapath = config['DATASET']['path']
    imgfiles = glob(os.path.join(datapath, "*.png"), recursive=True)

    dataset = CustomDataset(datapack=imgfiles, img_size=config['DATASET']['img_size'])
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True)

    criterion = nn.BCELoss()    # Binary Cross Entropy Loss (h(x), y)

    d_optimizer = torch.optim.Adam(d.parameters(), lr=float(config['TRAIN']['lr']))
    g_optimizer = torch.optim.Adam(g.parameters(), lr=float(config['TRAIN']['lr']))

    train_d_loss, train_g_loss = [], []

    for epoch in range(0, epochs):
        
        d = d.train()
        g = g.train()

        print(f"\nEpoch : {epoch}/{epochs}")
        for step, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            data = data.reshape([batch_size, -1]).to(_device)
            d_output = d(data)

            true_target = torch.ones(batch_size, 1).to(_device)      # True  = 1
            fake_target = torch.zeros(batch_size, 1).to(_device)     # False = 0

            d_loss = criterion(d_output, true_target) + criterion(d(g.forward(generate_random_seed(batch_size, 100).to(_device))), fake_target)
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            g_loss = criterion(d(g(generate_random_seed(batch_size, 100).to(_device))), true_target)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            d_loss = d_loss.detach().cpu().numpy()
            g_loss = g_loss.detach().cpu().numpy()

            train_d_loss.append(d_loss)
            train_g_loss.append(g_loss)
    
    d_losses = get_losses_graph(x=[x for x in range(len(train_d_loss))], y=[train_d_loss], labels='D Loss', savename='D_loss')
    g_losses = get_losses_graph(x=[x for x in range(len(train_g_loss))], y=[train_g_loss], labels='G Loss', savename='G_loss')

    plt.figure(dpi=300)
    f, axarr = plt.subplots(2, 3, figsize=(16, 8))

    with torch.no_grad():
        g = g.eval()
        for i in range(2):
            for j in range(3):
                output = g.forward(generate_random_seed(1, 100))
                img = output.detach().numpy().reshape(28, 28)
                axarr[i, j].imshow(img, interpolation='none', cmap='Blues')

    plt.show()
    torch.save(g.state_dict(), "generator.pt")
    """
    # load
    
    g = Generator()
    g.load_state_dict(torch.load('generator.pt'))
    g.eval()
    """
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

    d_model = Discriminator()

    d_model = d_model.to(_device)

    print("===== Discriminator ====")
    print(d_model)

    datapath = config['DATASET']['path']
    imgfiles = glob(os.path.join(datapath, "*.png"))

    dataset = CustomDataset(datapack=imgfiles, img_size=config['DATASET']['img_size'])
    dataloader = DataLoader(dataset=dataset, batch_size=config['TRAIN']['batch_size'], drop_last=True)

    criterion = nn.BCELoss()    # Binary Cross Entropy Loss (h(x), y)

    d_optimizer = torch.optim.Adam(d_model.parameters(), lr=float(config['TRAIN']['lr']))

    total_train_iter = len(dataloader)
    d_loss_list, g_loss_list = [], []
    epochs = int(config['TRAIN']['epochs'])

    for epoch in range(epochs):
        epoch_start = time.time()
        train_d_loss = []

        d_model = d_model.train()

        for step, data in enumerate(dataloader):
            iter_start = time.time()
            data = data.reshape([config['TRAIN']['batch_size'], -1])
            data = data.to(_device)

            true_target = torch.ones(config['TRAIN']['batch_size'], 1).to(_device)      # True  = 1
            fake_target = torch.zeros(config['TRAIN']['batch_size'], 1).to(_device)     # False = 0

            d_output_real = d_model(data)
            d_output_fake = d_model(generate_random(config['TRAIN']['batch_size'], 784))

            d_loss = criterion(d_output_real, true_target) + criterion(d_output_fake, fake_target)

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            d_loss = d_loss.detach().cpu().numpy()

            train_d_loss.append(d_loss)
            
            if step % 100 == 0:
                print("[train %s/%3s] Epoch: %3s | Time: %6.2fs/it | discriminator_loss: %6.4f" % (
                        step + 1, total_train_iter, epoch + 1, time.time() - iter_start, np.round(d_loss, 2)))
        
        train_d_loss = np.round(sum(train_d_loss) / total_train_iter, 2)

        d_loss_list.append(train_d_loss)

        print("\n[Epoch {} training Ended] > Time: {:.2}s/epoch | Discriminator Loss: {:.4f}\n".format(
                epoch + 1, time.time() - epoch_start, train_d_loss))

        get_losses_graph(x=[x for x in range(len(d_loss_list))], y=[d_loss_list], labels=['Discriminator Loss'])
        
        target_real = read_inference_images(shuffle(glob('/Users/bolero/dc/dataset/mnist_png/testing/8/*.png'))[0:4])
        target_random = [generate_random(1, 784) for x in range(0, 4)]

        for elems in zip(target_real, target_random):
            realval, randval = elems

            output_real = d_model.forward(realval)
            output_fake = d_model.forward(randval)

            print(f"Output Real : {output_real} | Output Random : {output_fake}\n\n")

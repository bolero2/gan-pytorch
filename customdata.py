import torch
from torch.utils.data import Dataset
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, datapack, img_size):
        super(CustomDataset).__init__()

        self.datapack = datapack
        self.img_size = img_size

    def __len__(self):
        return len(datapack)

    def __getitem__(self, index):
        self.index = index
        imgpath = self.datapack[index]

        img = Image.open(imgpath).convert('RGB')
        img = img.resize(self.img_size)


        


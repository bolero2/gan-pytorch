import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms 


def tensor_transform(tensor, img_size, mean, std):
    tensor = tensor.unsqueeze(0)
    tensor = tensor.type(torch.float32) 

    _transforms = torch.nn.Sequential(
        transforms.Normalize(mean, std),
    )

    # tensor = _transforms(tensor)
    tensor = tensor.reshape([-1, img_size[0] * img_size[1]])
    return tensor


class CustomDataset(Dataset):
    def __init__(self, datapack, img_size):
        super(CustomDataset, self).__init__()
        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]

        self.datapack = datapack
        self.img_size = img_size

    def __len__(self):
        return len(self.datapack)

    def __getitem__(self, index):
        self.index = index
        imgpath = self.datapack[index]

        img = Image.open(imgpath).convert('L')
        img = img.resize(self.img_size)
        img = np.array(img)
        img = img / 255.0
        img = torch.from_numpy(img)

        img = tensor_transform(img, self.img_size, self.mean, self.std)

        return img

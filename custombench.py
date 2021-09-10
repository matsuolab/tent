import os
import pathlib
import torch
import torchvision
import requests
import tarfile
import multiprocessing
import collections

from torchvision import transforms
from robustbench.data import CORRUPTIONS
from typing import Optional, Sequence, Tuple

import time
import torch.nn as nn
from collections import OrderedDict
from torch import Tensor


def _download_dataset(imagenetc_dl_path: str):
    file_names = [
        'blur.tar',
        'digital.tar',
        'extra.tar',
        'noise.tar',
        'weather.tar',]
    
    for file_name in file_names:
        url = f'https://zenodo.org/record/2235448/files/{file_name}?download=1'
        print(f'Downloading {url} ...')
        r = requests.get(url)
        with open(file_name , 'wb') as f:
            f.write(r.content)
        
        print(f'Extracting {file_name} ...')
        tar = tarfile.open(file_name)
        tar.extractall(path=imagenetc_dl_path)
        tar.close()



def load_imagenetc(n_examples: int,
                   severity: int, 
                   data_path: str,
                   corruption_name: str, 
                   minibatch_size: int, 
                   shuffle: bool = True, 
                   drop_last: bool = True):
    
    imagenetc_path = os.path.join(data_path, 'imagenet2012', 'val_c', corruption_name, str(severity))
    imagenetc_dl_path = os.path.join(data_path, 'imagenet2012', 'val_c')

    if not os.path.exists(imagenetc_dl_path):
        print(f'Creating directory {imagenetc_dl_path}')
        pathlib.Path(imagenetc_dl_path).mkdir(parents=True, exist_ok=True)
        _download_dataset(imagenetc_dl_path)
    else:
        print(f'Directory {imagenetc_dl_path} exists. Assuming dataset is already downloaded.')
    
    # Res256Crop224
    transform_without_da_imagenet = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    imagenet_c_testset = torchvision.datasets.ImageFolder(root=imagenetc_path,
                                                          transform=transform_without_da_imagenet)
    
    dataloader_num_workers = 4 # multiprocessing.cpu_count()

    imagenet_c_test_loader = torch.utils.data.DataLoader(
        imagenet_c_testset,
        shuffle=shuffle,
        batch_size=minibatch_size,
        drop_last=drop_last,
        num_workers=dataloader_num_workers,
        pin_memory=True)
    
    x_list, y_list = [], []
    for x, y in imagenet_c_test_loader:
        x_list.append(x)
        y_list.append(y)

    x = torch.cat(x_list, 0)
    y = torch.cat(y_list, 0)

    x = x[:n_examples]
    y = y[:n_examples]

    return x, y


# See https://github.com/RobustBench/robustbench/blob/5e8980aeb97f04a950ef128a890f8cb45f142f9b/robustbench/model_zoo/architectures/utils_architectures.py
class ImageNormalizer(nn.Module):
    def __init__(self, mean: Tuple[float, float, float],
        std: Tuple[float, float, float]) -> None:
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input: Tensor) -> Tensor:
        return (input - self.mean) / self.std


def normalize_model(model: nn.Module, mean: Tuple[float, float, float],
    std: Tuple[float, float, float]) -> nn.Module:
    layers = OrderedDict([
        ('normalize', ImageNormalizer(mean, std)),
        ('model', model)
    ])
    return nn.Sequential(layers)       


if __name__ == '__main__':

    minibatch_size_test = 32
    ga_steps_test =  1
    
    start = time.time()
    minibatch_size = int(minibatch_size_test / ga_steps_test)
    X, y = load_imagenetc(n_examples=10000,
                          severity=5,
                          data_path='./data',
                          corruption_name='motion_blur',
                          minibatch_size=minibatch_size)
    end = time.time()
    print(f"Elapsed (minbatch={minibatch_size})= {end - start}")
    print(f"X shape={X.shape} y shape={y.shape}")

    

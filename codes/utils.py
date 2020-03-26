'''
TODO:
    * all images are jpg
    * add Echo stuffs
'''

import os
import copy
import logging
from PIL import Image

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models



class SkidSteerDataset(Dataset):
    """Corrosion Detection dataset."""

    def __init__(self, csv_file, img_root, col_ids, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_root (string): Directory with all the images.
            col_ids (list): Indexes of column used in further modeling.
            transform (callable, optional): Optional transform to be applied on a sample (including augmentation).
        """
        self.csv_file = pd.read_csv(csv_file, index_col=0)
        self.img_root = img_root
        self.col_ids = col_ids
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        '''Return one data point with a PIL image and its label.'''
        img_dir = os.path.join(self.img_root, self.csv_file.iloc[idx, 0]) + ".jpg"
        price = self.csv_file.iloc[idx, 1]
        image = Image.open(img_dir)
        ftrs = torch.tensor(self.csv_file.iloc[idx, self.col_ids])  # change this when new columns are added
        
        if self.transform:
            image = self.transform(image)
        sample = {'price': price, 'image': image, "ftrs": ftrs}
        return sample


class PriceModel(nn.Module):
    '''Neural Network for price prediction.'''

    def __init__(self, num_ftrs=0, hidden_units=None, fine_tune=2):
        '''
            num_ftrs (int): Number of features added into this model.
            hidden_units (list): Number of neurons used in each fully connected layers.
            fine_tune (int): Number of CNN blocks (counting backwards) to fine tune.

        '''
        super().__init__()
        assert type(fine_tune) == int and fine_tune < 5, "fine_tune should be a non-negative int smaller than 5"

        cnn = models.resnet152(pretrained=True)
        for param in cnn.parameters():
            param.requires_grad = False
        if fine_tune > 0:
            for param in cnn.layer4.parameters():
                param.requires_grad = True
        if fine_tune > 1:
            for param in cnn.layer3.parameters():
                param.requires_grad = True
        if fine_tune > 2:
            for param in cnn.layer2.parameters():
                param.requires_grad = True
        if fine_tune > 3:
            for param in cnn.layer1.parameters():
                param.requires_grad = True
        cnn.fc = nn.Identity()

        fc_layers = nn.Sequential()
        if not hidden_units:
            fc_layers.add_module("fc1", nn.Linear(2048+num_ftrs, 1))
        else:
            linear_units = [2048+num_ftrs] + hidden_units
            for ii, (ins, outs) in enumerate(zip(linear_units[:-1], linear_units[1:])):
                fc_layers.add_module("fc"+str(ii+1), nn.Linear(ins, outs))
                fc_layers.add_module("activ"+str(ii+1), nn.Sigmoid())
            fc_layers.add_module("fc"+str(ii+2), nn.Linear(outs, 1))

        self.cnn = cnn
        self.fc_layers = fc_layers
        self.num_ftrs = num_ftrs
        self.fint_tune = fine_tune
        self.hidden_units = hidden_units

    def forward(self, images, ftrs):
        z = self.cnn(images)
        z = torch.cat([z, ftrs], dim=1)
        out = self.fc_layers(z)
        return out


def compute_price_loss(outputs, prices, min_max_scaler):
    '''Convert outputs to original scale and compute price loss.'''
    outputs = norm2price(outputs, min_max_scaler)
    prices = norm2price(prices, min_max_scaler)
    mae = np.abs(outputs - prices)
    maep = np.abs(outputs - prices) / prices
    return mae.sum(), mae, maep.sum(), maep


def norm2price(tensor, min_max_scaler):
    '''Convert tensor to its original scale.'''
    array2d = tensor.to("cpu").data.numpy().reshape(-1, 1)
    return np.exp(min_max_scaler.inverse_transform(array2d))
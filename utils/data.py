import h5py
from numpy import *
# from matplotlib.pyplot import *
import torch
import torch.utils.data
from torch import nn, optim
from tqdm import tqdm
# import matplotlib.pyplot as plt
import numpy as np


class Qubit_Readout_Dataset():
    
    def __init__(self, data, labels, sr):
        self.data = data
        self.labels = labels    
        self.data = self.data.reshape(len(data),sr*2)
    
    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_dataset():
    # Load the raw data
    IF = -136.75/1e3
    with h5py.File(r'../../datasets/qubits/00002_IQ_plot_raw.h5', 'r') as f:
        adc_g_1 = array(f['adc_g_1'])[0]
        adc_g_2 = array(f['adc_g_2'])[0]
        adc_e_1 = array(f['adc_e_1'])[0]
        adc_e_2 = array(f['adc_e_2'])[0]


    """ Select the range of time series data. Each data is 2000 element vector 
    representing 2000ns readout signal"""
    csr = range(500,1500)
    sr = len(csr)

    I_g = adc_g_1[:,csr]
    Q_g = adc_g_2[:,csr] 
    I_e = adc_e_1[:,csr] 
    Q_e = adc_e_2[:,csr] 

    # Dataset Creation
    data = zeros((adc_g_1.shape[0]*2,sr,2))
    data[0:adc_g_1.shape[0],:,0] = I_g
    data[0:adc_g_1.shape[0],:,1] = Q_g
    data[adc_g_1.shape[0]:adc_g_1.shape[0]*2,:,0] = I_e
    data[adc_g_1.shape[0]:adc_g_1.shape[0]*2,:,1] = Q_e

    labels = zeros(I_e.shape[0]*2)
    labels[I_e.shape[0]:I_e.shape[0]*2] = 1

    data = torch.from_numpy(data).float()
    labels = torch.from_numpy(labels).float()

    # Hyper-Parameters
    torch.manual_seed(4)
    train = 9000#data.shape[0]*0.99
    test = len(data)-train

    # Dataloader Prep
    dataset = Qubit_Readout_Dataset(data, labels, sr)
    train_data, test_data = torch.utils.data.random_split(dataset, [int(train), int(test)])
    return train_data, test_data


def get_dataloaders():
    num_workers = 0
    batch_size = 12800

    train_data, test_data = get_dataset()
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, 
                                                num_workers = num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, 
                                                num_workers = num_workers, shuffle=True)
    return train_loader, test_loader

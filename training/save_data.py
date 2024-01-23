from __future__ import print_function
import h5py
from numpy import *
from matplotlib.pyplot import *
import torch
import torch.utils.data
from torch import nn, optim
from torch.utils.data.dataset import ConcatDataset, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torchinfo 
import argparse 
import os 

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data-dir', type=str, default='../data/')
args = parser.parse_args()

# Hyper-Parameters
torch.manual_seed(4)
batch_size = 12800
device = torch.device("cpu")
kwargs = {'num_workers':0, 'pin_memory': True} 

### Load and Prepare the Data ###

# Load the raw data
IF = -136.75/1e3
csr = range(335-50,335+50)
csr = range(770)
sr = len(csr)

#######################################
# Dataset class template 
#######################################
class Qubit_Readout_Dataset(torch.utils.data.Dataset):
    
    def __init__(self, file_path):
        with h5py.File(file_path, 'r') as f:
            g_data = np.array(f['g_data'])[0]
            e_data = np.array(f['e_data'])[0]
            start_stamp = np.array(f['start_stamp'])
            end_stamp = np.array(f['end_stamp'])

        adc_g_1 = g_data[:,0,:] 
        adc_g_2 = g_data[:,1,:]
        adc_e_1 = e_data[:,0,:]
        adc_e_2 = e_data[:,1,:]

        I_g = adc_g_1[:,csr]
        Q_g = adc_g_2[:,csr] 
        I_e = adc_e_1[:,csr] 
        Q_e = adc_e_2[:,csr] 

        self.data = zeros((adc_g_1.shape[0]*2,sr,2))
        self.data[0:adc_g_1.shape[0],:,0] = I_g
        self.data[0:adc_g_1.shape[0],:,1] = Q_g
        self.data[adc_g_1.shape[0]:adc_g_1.shape[0]*2,:,0] = I_e
        self.data[adc_g_1.shape[0]:adc_g_1.shape[0]*2,:,1] = Q_e

        labels = zeros(I_e.shape[0]*2)
        labels[I_e.shape[0]:I_e.shape[0]*2] = 1

        self.data = torch.from_numpy(self.data).float()
        self.labels = torch.from_numpy(labels).float()

        self.data = self.data.reshape(len(self.data), sr*2)
       
    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def __add__(self, other: Dataset) -> ConcatDataset:
        self.data = torch.cat((self.data, other.data), 0)
        self.labels = torch.cat((self.labels, other.labels), 0)
        return self


#######################################
# Load dataset 
#######################################
total_samples = 0
dataset = None

# Data loop
for i in tqdm(range(0, 101)): # Loop over files
    file_name = f'{args.data_dir}/{str(i).zfill(5)}_ge_RAW_ADC.h5' # Generates 00000, '00001', '00002', ..., '00100'

    if dataset:
        dataset += Qubit_Readout_Dataset(file_name)
    else: 
        dataset = Qubit_Readout_Dataset(file_name)


#######################################
# Partition into train-test split 
#######################################
# Indices
all_indices = np.arange(len(dataset))
# np.random.shuffle(all_indices)

train_indices = all_indices[:int(0.9*len(all_indices))]
test_indices = all_indices[int(0.9*len(all_indices)):]

train_data = torch.utils.data.Subset(dataset, train_indices)
test_data = torch.utils.data.Subset(dataset, test_indices)

#######################################
# Create train-test dataloaders
#######################################
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, **kwargs)

#######################################
# Digitize into npy arrays 
#######################################
X_train, y_train = train_data[:]
X_test, y_test = test_data[:]


np.save(os.path.join(args.data_dir, 'X_train.npy'), X_train)
np.save(os.path.join(args.data_dir, 'y_train.npy'), y_train)
np.save(os.path.join(args.data_dir, 'X_test.npy'), X_test)
np.save(os.path.join(args.data_dir, 'y_test.npy'), y_test)

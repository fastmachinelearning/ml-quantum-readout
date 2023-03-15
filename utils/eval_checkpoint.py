from __future__ import print_function
import h5py
from numpy import *
from matplotlib.pyplot import *
import torch
import torch.utils.data
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

### Load and Prepare the Data ###
from data import test_loader

# Load the raw data
# IF = -136.75/1e3
# with h5py.File(r'../../datasets/qubits/00002_IQ_plot_raw.h5', 'r') as f:
#     adc_g_1 = array(f['adc_g_1'])[0]
#     adc_g_2 = array(f['adc_g_2'])[0]
#     adc_e_1 = array(f['adc_e_1'])[0]
#     adc_e_2 = array(f['adc_e_2'])[0]

# # For plotting raw data    
# # plt.figure()
# # plt.plot(adc_g_1[0,:],label= "g - In-phase(I)")
# # plt.legend()
# # plt.figure()
# # plt.plot(adc_g_2[0,:],label= "g - Quadrature(Q)")
# # plt.legend()
# # plt.figure()
# # plt.plot(adc_e_1[0,:],label= "e - In-phase(I)")
# # plt.figure()
# # plt.plot(adc_e_2[0,:],label= "e - Quadrature(Q)")

# """ Select the range of time series data. Each data is 2000 element vector 
# representing 2000ns readout signal"""
# csr = range(0,2000)
# sr = len(csr)

# I_g = adc_g_1[:,csr]
# Q_g = adc_g_2[:,csr] 
# I_e = adc_e_1[:,csr] 
# Q_e = adc_e_2[:,csr] 

# # Dataset Creation
# data = zeros((adc_g_1.shape[0]*2,sr,2))
# data[0:adc_g_1.shape[0],:,0] = I_g
# data[0:adc_g_1.shape[0],:,1] = Q_g
# data[adc_g_1.shape[0]:adc_g_1.shape[0]*2,:,0] = I_e
# data[adc_g_1.shape[0]:adc_g_1.shape[0]*2,:,1] = Q_e

# labels = zeros(I_e.shape[0]*2)
# labels[I_e.shape[0]:I_e.shape[0]*2] = 1

# data = torch.from_numpy(data).float()
# labels = torch.from_numpy(labels).float()

# class Qubit_Readout_Dataset():
    
#     def __init__(self):
#         self.data = data
#         self.labels = labels    
#         self.data = self.data.reshape(len(data),sr*2)
       
#     def __len__(self):
#         return self.labels.shape[0]

#     def __getitem__(self, idx):
#         return self.data[idx], self.labels[idx]

# # Hyper-Parameters
# torch.manual_seed(4)
# epochs = 10
# train = 9000#data.shape[0]*0.99
# test = len(data)-train
# batch_size = 12800
learning_rate = 1e-4

# # device = torch.device("cuda")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# kwargs = {'num_workers':0, 'pin_memory': True} 
# plot_interval = 1

# # Dataloader Prep
# dataset = Qubit_Readout_Dataset()
# train_data, test_data = torch.utils.data.random_split(dataset, [int(train), int(test)])

# num_workers = 0
# train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, 
#                                             num_workers = num_workers, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, 
#                                             num_workers = num_workers, shuffle=True)
    
# NN initialization
# class Classifier(nn.Module):
#     def __init__(self):
#         super(Classifier, self).__init__()

#         self.hn = sr*2 * 4
#         self.classifier = nn.Sequential(nn.Linear(sr*2, self.hn),
#                                     nn.ReLU(inplace=True),
#                                     nn.Linear(self.hn, self.hn),
#                                     nn.ReLU(inplace=True),
#                                     nn.Dropout(p=0.8),
#                                     nn.Linear(self.hn, int(self.hn/8)),
#                                     nn.ReLU(inplace=True),
#                                     nn.BatchNorm1d(int(self.hn/8),affine=False),
#                                     nn.Dropout(p=0.8),
#                                     nn.Linear(int(self.hn/8), 2),
#                                     nn.ReLU(inplace=True),)

#     def forward(self, sig):
#         state = self.classifier(sig)
#         return state
csr = range(500, 1500)
sr = len(csr)
hn = sr * 2 * 1

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.linear1 = nn.Linear(sr * 2, int(hn / 8))
        self.relu1 = nn.ReLU()
        self.bn = nn.BatchNorm1d(int(hn / 8), affine=True)

        self.linear2 = nn.Linear(int(hn / 8), 2)
        self.relu2 = nn.ReLU()

    def forward(self, sig):
        x = self.linear1(sig)
        x = self.relu1(x)
        x = self.bn(x)

        x = self.linear2(x)
        x = self.relu2(x)
        return x


# NN Model and the optimizier 
model = Classifier().to(device)
print(model)
model.load_state_dict(torch.load('../checkpoints/checkpoint_tiny_affine.pth'))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Readout Fidelity
model.eval() 
cc = 0
y_true = torch.tensor([]).to(device)
y_pred = torch.tensor([]).to(device)

with torch.no_grad():
    for data, target in test_loader:
        
        data=data.to(device)
        states = model(data)
        target = target.to(device) 

        val, ind = torch.max(states,1)
        y_pred = torch.cat((y_pred, ind), 0)
        y_true = torch.cat((y_true, target), 0)

acc = y_true-y_pred
accuracy = (len(y_true)-torch.count_nonzero(acc))/len(y_true)
accuracy = accuracy.item()

print('Readout Fidelity: %', accuracy*100)

num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Model Parameters: %', num_parameters)

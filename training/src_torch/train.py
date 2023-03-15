from __future__ import print_function

import sys
sys.path.append("..")

from numpy import *
from matplotlib.pyplot import *
import torch
import torch.utils.data
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from utils.data import get_dataloaders
from models.stream_test import *
from models.tiny import *


# Hyper-Parameters
torch.manual_seed(4)
epochs = 500
batch_size = 256
learning_rate = 1e-4

device = torch.device("cpu")
plot_interval = 1

# dataloaders 
train_loader, test_loader = get_dataloaders()

# NN Model and the optimizier 
model = TinyClassifierv2(1000).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Loss tracks
train_loss_track=np.array([])
test_loss_track=np.array([])
acc_track=np.array([])

# Training loop
for epoch in tqdm(range(epochs)):
    
    train_loss = 0
    model.train() 
    
    for data, labels in train_loader:
        optimizer.zero_grad()
        data, labels = data.to(device), labels.to(device)
        states = model(data)
        loss = criterion(states, labels.long())
        loss.backward()
        train_loss += loss.detach().cpu().numpy()
        optimizer.step()
        
    train_loss_track = np.append(train_loss_track,np.asarray(train_loss))
        
    test_loss = 0    
    model.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            print(data.shape)
            states = model(data)
            loss = criterion(states, labels.long())
            test_loss += loss.detach().cpu().numpy()
    
    test_loss_track = np.append(test_loss_track,np.asarray(test_loss))
    
    if epoch % plot_interval == 0:
        print('====> Epoch: {} Training loss: {:.6f}'.format(
                  epoch, train_loss ))
        print('====> Epoch: {} Test loss: {:.6f}'.format(
                  epoch, test_loss ))

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
    
    acc_track = np.append(acc_track,np.asarray(accuracy))
    print('Readout Fidelity: %', accuracy*100)

print('Saving Model State...')
torch.save(model.state_dict(), '../checkpoints/tinyv2.pth')

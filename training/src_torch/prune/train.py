import sys
sys.path.append("../..")

import torch 
import torch.nn as nn
import torch.nn.utils.prune as prune

import torch.utils.data
from torch import nn, optim
from tqdm import tqdm
import numpy as np

from utils.data import get_dataloaders


class Net(nn.Module):
    def __init__(self, sr):
        super(Net, self).__init__()

        self.hn = sr*2 * 1
        
        self.linear1 = nn.Linear(sr*2, int(self.hn/8))
        self.relu1 = nn.ReLU()
        self.bn = nn.BatchNorm1d(int(self.hn/8),affine=True)

        self.linear2 = nn.Linear(int(self.hn/8), 2)
        self.relu2 = nn.ReLU()

    def forward(self, sig):
        x = self.linear1(sig)
        x = self.relu1(x)
        x = self.bn(x)

        x = self.linear2(x)
        x = self.relu2(x)
        return x
    
model = Net(1000)
for name, module in model.named_modules():
    # prune 20% of connections in all 2D-conv layers
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.95)
    # prune 40% of connections in all linear layers
    elif isinstance(module, torch.nn.Linear):
        if name == 'linear1':
            prune.l1_unstructured(module, name='weight', amount=0.95)
        else:
            prune.l1_unstructured(module, name='weight', amount=0.85)

print(dict(model.named_buffers()).keys())  # to verify that all masks exist


# Hyper-Parameters
torch.manual_seed(4)
epochs = 500
batch_size = 512
learning_rate = 1e-3

device = torch.device("cpu")
plot_interval = 1

# dataloaders 
train_loader, test_loader = get_dataloaders()

# NN Model and the optimizier 
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
torch.save(model.state_dict(), '../../checkpoints/prune_model_p95.pth')

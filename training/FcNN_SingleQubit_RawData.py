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

# Load the raw data
IF = -136.75/1e3
with h5py.File(r'../datasets/qubits/00002_IQ_plot_raw.h5', 'r') as f:
    adc_g_1 = array(f['adc_g_1'])[0]
    adc_g_2 = array(f['adc_g_2'])[0]
    adc_e_1 = array(f['adc_e_1'])[0]
    adc_e_2 = array(f['adc_e_2'])[0]

# For plotting raw data    
plt.figure()
plt.plot(adc_g_1[0,:],label= "g - In-phase(I)")
plt.legend()
plt.savefig('g - In-phase(I).png')
plt.figure()
plt.plot(adc_g_2[0,:],label= "g - Quadrature(Q)")
plt.legend()
plt.savefig('g - Quadrature(Q).png')

plt.figure()
plt.plot(adc_e_1[0,:],label= "e - In-phase(I)")
plt.legend()
plt.savefig('e - In-phase(I).png')

plt.figure()
plt.plot(adc_e_2[0,:],label= "e - Quadrature(Q)")
plt.legend()
plt.savefig('e - Quadrature(Q).png')

""" Select the range of time series data. Each data is 2000 element vector 
representing 2000ns readout signal"""
csr = range(0,2000)
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

class Qubit_Readout_Dataset():
    
    def __init__(self):
        self.data = data
        self.labels = labels    
        self.data = self.data.reshape(len(data),sr*2)
       
    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Hyper-Parameters
torch.manual_seed(4)
epochs = 25
train = 9000#data.shape[0]*0.99
test = len(data)-train
batch_size = 12800
learning_rate = 1e-4

# device = torch.device("cuda")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kwargs = {'num_workers':0, 'pin_memory': True} 
plot_interval = 1

# Dataloader Prep
dataset = Qubit_Readout_Dataset()
train_data, test_data = torch.utils.data.random_split(dataset, [int(train), int(test)])

num_workers = 0
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, 
                                            num_workers = num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, 
                                            num_workers = num_workers, shuffle=True)
    
# NN initialization
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.hn = sr*2 * 4
        self.classifier = nn.Sequential(nn.Linear(sr*2, self.hn),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(self.hn, self.hn),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=0.8),
                                    nn.Linear(self.hn, int(self.hn/8)),
                                    nn.ReLU(inplace=True),
                                    nn.BatchNorm1d(int(self.hn/8),affine=False),
                                    nn.Dropout(p=0.8),
                                    nn.Linear(int(self.hn/8), 2),
                                    nn.ReLU(inplace=True),)

    def forward(self, sig):
        state = self.classifier(sig)
        return state

# NN Model and the optimizier 
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Loss tracks
train_loss_track=np.array([])
test_loss_track=np.array([])

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
            states = model(data)
            loss = criterion(states, labels.long())
            test_loss += loss.detach().cpu().numpy()
    
    test_loss_track = np.append(test_loss_track,np.asarray(test_loss))
    
    if epoch % plot_interval == 0:
        print('====> Epoch: {} Training loss: {:.6f}'.format(
                  epoch, train_loss ))
        print('====> Epoch: {} Test loss: {:.6f}'.format(
                  epoch, test_loss ))

        p1, p2 = 0, 800
        plt.plot(train_loss_track[p1:p2],label = 'Training')
        plt.plot(test_loss_track[p1:p2],label='Testing')
        plt.legend()
        plt.xlabel('Epochs')
        plt.title("Training - Test Loss")
        plt.figure(figsize = (12,7))
        plt.show()

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

print('Saving Model State...')
torch.save(model.state_dict(), '../checkpoints/checkpoint_baseline.pth')


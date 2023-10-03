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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int)
parser.add_argument('--window', type=int)
args = parser.parse_args()


# Hyper-Parameters
torch.manual_seed(4)
epochs = 10
batch_size = 12800
learning_rate = 1e-2

device = torch.device("cpu")
kwargs = {'num_workers':0, 'pin_memory': True} 
plot_interval = 1

### Load and Prepare the Data ###

# Load the raw data
IF = -136.75/1e3
# csr = range(335-50,335+50)  # ORIGINAL 
csr = range(args.start, args.start+args.window)
sr = len(csr)

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

# NN initialization
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.hn = sr*2 * 1
        self.classifier = nn.Sequential(
            nn.Linear(sr*2, int(self.hn/8)),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(int(self.hn/8),affine=False),
            nn.Linear(int(self.hn/8), 2),
            nn.ReLU(inplace=True)
        )
        # self.classifier = nn.Sequential(
        #     nn.Linear(sr*2, 2),
        #     nn.BatchNorm1d(2,affine=False)
        # )

    def forward(self, sig):
        state = self.classifier(sig)
        return state

# NN Model and the optimizier 
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

print('###################################')
print(model)
print('###################################')
import torchinfo
torchinfo.summary(model, input_size=(1, sr*2))
print('###################################')

# Indices
all_indices = np.arange(10000)
np.random.shuffle(all_indices)

train_indices = all_indices[:int(0.8*len(all_indices))]
test_indices = all_indices[int(0.8*len(all_indices)):]

# Loss tracks
train_loss_track=np.array([])
test_loss_track=np.array([])
acc_track=np.array([])
acc_history=np.array([])
# Training loop
for epoch in tqdm(range(epochs)):
    acc_epoch = [] # List to store accuracies of all files in this epoch
    
    for i in tqdm(range(1, 101)): # Loop over files
        file_name = f'../../data/new-raw-data/{str(i).zfill(5)}_ge_RAW_ADC.h5' # Generates '00001', '00002', ..., '00100'

        dataset = Qubit_Readout_Dataset(file_name)
        
        # train_data, test_data = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), int(0.2*len(dataset))])
        train_data = torch.utils.data.Subset(dataset, train_indices)
        test_data = torch.utils.data.Subset(dataset, test_indices)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, **kwargs)

        # Training
        model.train() 
        train_loss = 0
        for data, labels in train_loader:
            optimizer.zero_grad()
            data, labels = data.to(device), labels.to(device)
            states = model(data)
            loss = criterion(states, labels.long())
            loss.backward()
            train_loss += loss.detach().cpu().numpy()
            optimizer.step()
        
        train_loss_track = np.append(train_loss_track,np.asarray(train_loss))

        # Testing
        model.eval()
        test_loss = 0 
        y_true = torch.tensor([]).to(device)
        y_pred = torch.tensor([]).to(device)
        
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                states = model(data)
                loss = criterion(states, labels.long())
                test_loss += loss.detach().cpu().numpy()

                # Readout Fidelity
                val, ind = torch.max(states,1)
                y_pred = torch.cat((y_pred, ind), 0)
                y_true = torch.cat((y_true, labels), 0)

            acc = y_true - y_pred
            accuracy = (len(y_true) - torch.count_nonzero(acc)) / len(y_true)
            accuracy = accuracy.item()

            # Add the accuracy of this file to the list for this epoch
            acc_epoch.append(accuracy)
            
            acc_track = np.append(acc_track, np.asarray(accuracy))
    
        test_loss_track = np.append(test_loss_track,np.asarray(test_loss))

        print('====> Epoch: {} File: {} Training loss: {:.6f}'.format(epoch, i, train_loss))
        print('====> Epoch: {} File: {} Test loss: {:.6f}'.format(epoch, i, test_loss))

    # Calculate and print the average accuracy for this epoch
    avg_accuracy = np.mean(acc_epoch)
    print('====> Epoch: {} Average Readout Fidelity: {:.2f}%'.format(epoch, avg_accuracy * 100))
    
    # Update accuracy history with the average accuracy for this epoch
    acc_history = np.append(acc_history, np.asarray(avg_accuracy))
    
    # Print the maximum accuracy so far
    print('====> Epoch: {} Maximum Readout Fidelity: {:.2f}%'.format(epoch, np.max(acc_history) * 100))

p1, p2 = 0, -1
plt.figure(0)
plt.plot(train_loss_track[p1:p2],label = 'Training')
plt.plot(test_loss_track[p1:p2],label='Testing')
plt.legend()
plt.xlabel('Epochs')
plt.title("Training - Test Loss")
plt.figure(figsize = (12,7))
# plt.show()
# plt.savefig(f'results/test-loss_s{args.start}_w{args.window}.png')
plt.close()

f = open(f"../results/log-{args.window}.txt", "a")
f.write(f"{args.start}-{args.window}-{avg_accuracy}\n")
f.close()

"""
Digitizes data into numpy arrays. 

Usage:
    python save_data.py --start-window 0 --end-window 770 --data-dir ../data
"""
from __future__ import print_function
import os
import argparse
from tqdm import tqdm
import numpy as np
import h5py


#######################################
# Dataset class template 
#######################################
class Qubit_Readout_Dataset(object):
    
    def __init__(self, file_path, csr, sr):
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

        self.data = np.zeros((adc_g_1.shape[0]*2,sr,2))
        self.data[0:adc_g_1.shape[0],:,0] = I_g
        self.data[0:adc_g_1.shape[0],:,1] = Q_g
        self.data[adc_g_1.shape[0]:adc_g_1.shape[0]*2,:,0] = I_e
        self.data[adc_g_1.shape[0]:adc_g_1.shape[0]*2,:,1] = Q_e

        self.labels = np.zeros(I_e.shape[0]*2)
        self.labels[I_e.shape[0]:I_e.shape[0]*2] = 1
        self.data = self.data.reshape(len(self.data), sr*2)
       
    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def __add__(self, other):
        self.data = np.concatenate((self.data, other.data), 0)
        self.labels = np.concatenate((self.labels, other.labels), 0)
        return self


def process_data(start_window, end_window, data_dir):
    ### Load and Prepare the Data ###

    # Load the raw data
    csr = range(start_window, end_window)
    sr = len(csr)

    #######################################
    # Load dataset 
    #######################################
    dataset = None

    # Data loop
    for i in tqdm(range(0, 100)): # Loop over files
        file_name = f'{data_dir}/{str(i).zfill(5)}_ge_RAW_ADC.h5' # Generates 00000, '00001', '00002', ..., '00100'
        if dataset:
            dataset += Qubit_Readout_Dataset(file_name, csr, sr)
        else: 
            dataset = Qubit_Readout_Dataset(file_name, csr, sr)

    #######################################
    # Partition into train-test split 
    #######################################
    # Indices
    all_indices = np.arange(len(dataset))

    train_indices = all_indices[:int(0.9*len(all_indices))]
    test_indices = all_indices[int(0.9*len(all_indices)):]

    X_train, y_train = dataset[train_indices]
    X_test, y_test = dataset[test_indices]

    #######################################
    # Digitize into npy arrays 
    #######################################
    np.save(os.path.join(data_dir, f'X_train_{start_window}_{end_window}.npy'), X_train)
    np.save(os.path.join(data_dir, f'y_train_{start_window}_{end_window}.npy'), y_train)
    np.save(os.path.join(data_dir, f'X_test_{start_window}_{end_window}.npy'), X_test)
    np.save(os.path.join(data_dir, f'y_test_{start_window}_{end_window}.npy'), y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--start-window', type=int, default=0)
    parser.add_argument('-e', '--end-window', type=int, default=770)
    parser.add_argument('-d', '--data-dir', type=str, default='../data/')
    args = parser.parse_args()

    process_data(
        data_dir=args.data_dir,
        start_window=args.start_window,
        end_window=args.end_window,
    )

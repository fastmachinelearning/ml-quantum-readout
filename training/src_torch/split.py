import os
import argparse
from datetime import datetime
import numpy as np 
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import torch 
from torch import optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class Model(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.in_model_output_shape = 3
        self.num_models = args.num_models
        self.share = args.share
        self.models = self.get_models(args)

    def get_split_models(self, input_shape):
        nodes_fc1 = 25
        # model = nn.Sequential(
        #     nn.Linear(input_shape, nodes_fc1),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(nodes_fc1, self.in_model_output_shape),
        # )
        from models import Model1, QModel1
        model = QModel1(Model1(input_shape, nodes_fc1, self.in_model_output_shape))
        return model

    def get_joint_model(self, input_shape):
        # model = nn.Sequential(
        #     nn.Linear(input_shape*self.in_model_output_shape, 2),
        #     nn.Softmax(dim=1)
        # )
        from models import Model2, QModel2
        model = QModel2(Model2(input_shape*self.in_model_output_shape))
        return model

    def get_models(self, args):
        # self.input_shape = int(800 / args.num_models)
        self.input_shape = int(args.input_shape / args.num_models)

        models = torch.nn.ModuleList()
        for idx in range(args.num_models):
            if idx>0 and self.share:
                models.append(models[idx-1])
            else:
                models.append(self.get_split_models(self.input_shape))

        models.append(self.get_joint_model(args.num_models))

        return models

    def forward(self, x):
        outputs = torch.tensor([])

        for idx in range(self.num_models):
            output = self.models[idx](x[:, idx*self.input_shape:idx*self.input_shape+self.input_shape])
            outputs = torch.cat((outputs, output), 1) 

        outputs = self.models[-1](outputs)
        return outputs


def one_hot_encode(data):
    y_encoded = np.zeros([data.shape[0],2], dtype=np.int32)
    for idx, x in enumerate(data):
        if x == 1:
            y_encoded[idx][1] = 1
        else:
            y_encoded[idx][0] = 1
    return y_encoded


def load_data(args):
    X_train = np.load(os.path.join(args.data_dir, 'X_train_val.npy'))
    X_test = np.load(os.path.join(args.data_dir, 'X_test.npy'))    
    y_train = np.load(os.path.join(args.data_dir, 'y_train_val.npy'))
    y_test = np.load(os.path.join(args.data_dir, 'y_test.npy'), allow_pickle=True)
    
    print(args.data_dir)
    print(X_train.shape)
    args.input_shape = X_train.shape[1]

    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)

    # transform to torch tensor
    X_train = torch.Tensor(X_train) 
    y_train = torch.Tensor(y_train)
    X_test = torch.Tensor(X_test) 
    y_test = torch.Tensor(y_test)

    dataset = TensorDataset(X_train,y_train) # create your datset
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size) # create your dataloader

    dataset = TensorDataset(X_test,y_test) 
    test_dataloader = DataLoader(dataset, batch_size=args.batch_size)

    return train_dataloader, test_dataloader


def train(model, train_loader, test_loader, args):
    # input_shape = int(800 / args.num_models)
    input_shape = int(args.input_shape / args.num_models)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    bp = 0
    device = 'cpu'
    # Training loop
    for epoch in tqdm(range(args.epochs)):
        
        train_loss = 0
        model.train() 
        
        for data, labels in train_loader:
            optimizer.zero_grad()
            data, labels = data.to(device), labels.to(device)
            states = model(data)
            loss = criterion(states, labels.float())
            loss.backward()
            train_loss += loss.detach().cpu().numpy()
            optimizer.step()
            
        test_loss = 0    
        model.eval()
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                states = model(data)
                loss = criterion(states, labels.float())
                test_loss += loss.detach().cpu().numpy()
        
        
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
                val, ind = torch.max(target,1)
                y_true = torch.cat((y_true, ind), 0)
        
        acc = y_true-y_pred
        accuracy = (len(y_true)-torch.count_nonzero(acc))/len(y_true)
        accuracy = accuracy.item()
        
        print('Readout Fidelity: %', accuracy*100)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args):
    start_time = datetime.now()

    train_dataloader, test_dataloader = load_data(args)

    model = Model(args)
    train(model, train_dataloader, test_dataloader, args)

    print('-----------------------------------------------------')
    print(f'Input Length: {args.input_shape}')
    print(f'Number Models: {args.num_models}')
    print(f'Model Parameters: {count_parameters(model)}')
    print(f'Input Model Parameters: {count_parameters(model.models[0])}')
    print(f'Output Model Parameters: {count_parameters(model.models[-1])}')
    print('-----------------------------------------------------')
    print('Input Model')
    print(model.models[0])
    print('-----------------------------------------------------')
    print('Output Model')
    print(model.models[-1])
    print('-----------------------------------------------------')


    time_elapsed = datetime.now() - start_time
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

    # x_randn = torch.randn([1,160])
    # torch.onnx.export(model.models[0], x_randn, 'model1.onnx')
    # x_randn = torch.randn([1,15])
    # torch.onnx.export(model.models[-1], x_randn, 'model2.onnx')
    export_models = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training Options.')
    parser.add_argument('--num-models', type=int, default=5)
    # parser.add_argument('-d', '--data-dir', type=str, default='../../data/s5')
    parser.add_argument('-d', '--data-dir', type=str, default='../../data/all')
    # parser.add_argument('-d', '--data-dir', type=str, default='../../data/i0_w2000_s5')
    parser.add_argument('-b', '--batch-size', type=int, default=1024)
    parser.add_argument('-e', '--epochs',type=int, default=20)
    parser.add_argument('-v', '--val-split', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-q', '--quantize', action='store_true')
    parser.add_argument('-s', '--share', action='store_true')
    args = parser.parse_args()

    main(args)

# python split.py -d ../../data/s5/ -q

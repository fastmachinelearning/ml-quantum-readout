import torch.nn as nn


class TinyClassifier(nn.Module):
    def __init__(self, sr):
        super(TinyClassifier, self).__init__()

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


class PruneModel(nn.Module):
    def __init__(self, sr):
        super(PruneModel, self).__init__()

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


class TinyClassifierv2(nn.Module):
    def __init__(self, sr):
        super(TinyClassifierv2, self).__init__()

        self.hn = sr*2 * 1
        
        self.linear1 = nn.Linear(sr*2, 75)
        self.relu1 = nn.ReLU()
        self.bn = nn.BatchNorm1d(75,affine=True)

        self.linear2 = nn.Linear(75, 50)
        self.linear3 = nn.Linear(50, 25)
        self.linear4 = nn.Linear(25, 2)
        self.bn2 = nn.BatchNorm1d(50,affine=True)
        self.relu2 = nn.ReLU()

    def forward(self, sig):
        x = self.linear1(sig)
        # x = self.relu1(x)
        x = self.bn(x)

        x = self.linear2(x)
        x = self.bn2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        return x
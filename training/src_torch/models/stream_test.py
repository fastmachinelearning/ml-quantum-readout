import torch.nn as nn


class Classifierv1(nn.Module):
    def __init__(self):
        super(Classifierv1, self).__init__()
        
        self.linear1 = nn.Linear(2000, 1)
        ## Sigmoind function is not supported :(
        # self.sigmoid = nn.Sigmoid()

    def forward(self, sig):
        x = self.linear1(sig)
        return x


class Classifierv2(nn.Module):
    def __init__(self):
        super(Classifierv2, self).__init__()
        
        self.linear1 = nn.Linear(2000, 10)
        self.linear2 = nn.Linear(10, 1)

    def forward(self, sig):
        x = self.linear1(sig)
        x = self.linear2(x)
        return x

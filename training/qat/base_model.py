import torch.nn as nn

csr = range(500, 1500)
sr = len(csr)
hn = sr * 2 * 1


class TinyClassifier(nn.Module):
    def __init__(self):
        super(TinyClassifier, self).__init__()

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

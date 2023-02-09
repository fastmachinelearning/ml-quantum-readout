import torch 
import torch.nn as nn
import torch.nn.functional as F

import brevitas.nn as qnn
from brevitas.quant import Int32Bias, Int8ActPerTensorFloat


csr = range(500, 1500)
sr = len(csr)
hn = sr * 2 * 1


class BrevitasModel(nn.Module):
    def __init__(self):
        super(BrevitasModel, self).__init__()

        self.quant_inp = qnn.QuantIdentity(bit_width=12, return_quant_tensor=True)
        
        self.fc1   = qnn.QuantLinear(sr * 2, int(hn / 8), bias=True, weight_bit_width=6, bias_quant=Int32Bias)
        self.relu1 = qnn.QuantReLU(bit_width=12, return_quant_tensor=True)
        
        self.bn = nn.BatchNorm1d(int(hn / 8), affine=True)

        self.fc2   = qnn.QuantLinear(int(hn / 8), 2, bias=True, input_quant=Int8ActPerTensorFloat, weight_bit_width=6, bias_quant=Int32Bias)
        self.relu2 = qnn.QuantReLU(bit_width=12, return_quant_tensor=True)


    def forward(self, x):
        out = self.quant_inp(x)
        out = self.relu1(self.fc1(out))
        out = self.bn(out)
        out = self.relu2(self.fc2(out))
        return out


if __name__ == "__main__":
    model = BrevitasModel()
    print("====================================")
    print(model)
    print("====================================")

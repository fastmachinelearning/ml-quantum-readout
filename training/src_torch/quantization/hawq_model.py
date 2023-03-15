import torch.nn as nn

from hawq.utils.quantization_utils.quant_modules import QuantLinear, QuantAct


csr = range(500, 1500)
sr = len(csr)
hn = sr * 2 * 1


class HawqTinyClassifier(nn.Module):
    def __init__(self, model):
        super(HawqTinyClassifier, self).__init__()

        self.quant_input = QuantAct(activation_bit=5)
        self.q_relu1 = QuantAct(activation_bit=8)
        self.q_relu2 = QuantAct(activation_bit=8)

        layer = getattr(model, "linear1")
        hawq_layer = QuantLinear(weight_bit=2, bias_bit=2)
        hawq_layer.set_param(layer)
        setattr(self, "linear1", hawq_layer)

        layer = getattr(model, "linear2")
        hawq_layer = QuantLinear(weight_bit=2, bias_bit=2)
        hawq_layer.set_param(layer)
        setattr(self, "linear2", hawq_layer)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(100, affine=True)

    def forward(self, sig):
        x, p_sf = self.quant_input(sig)

        x = self.relu(self.linear1(x, p_sf))
        x, p_sf = self.q_relu1(x, p_sf)

        x = self.bn(x)
        
        x = self.relu(self.linear2(x, p_sf))
        x, p_sf = self.q_relu2(x, p_sf)
        return x

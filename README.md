# ml-quantum-readout
This project uses hls4ml to create the firmware implementation of machine learning algorithms for single and mult-qubit readout.

## Setup

Dependencies 
* Python==3.7
* PyTorch==1.12.0
* hls4ml==0.6.0

Clone repository
```
git clone --recursive https://github.com/jicampos/ml-quantum-readout.git
```

Create conda environment
```
conda env create
conda activate ml4qick-env
```

Single and mult-qubit data can be found [here](https://urldefense.proofpoint.com/v2/url?u=https-3A__purdue0-2Dmy.sharepoint.com_-3Af-3A_g_personal_oyesilyu-5Fpurdue-5Fedu_EuhbLM-2DwFApNiX9Mh5ZMeIEBG3dGqSIPgwN21j5S30nxvQ-3Fe-3DCDc3Xi&d=DwMFAg&c=gRgGjJ3BkIsb5y6s49QqsA&r=3tXuppM5Ux2UBnxU0DCrdSagIS9IpvGOlIFtsYfyWuc&m=5R-PzD5Udxkr2BBA9AYXREVhYselyKDYk_-1g6QMka_dPV3VTCVJe4id5PFOgpLq&s=fUu9yFLybrPN_AYcDhfBiQoXf5RlOAwbo6DmsD3CiqU&e=).

## Training 
A Multi-Layer Perceptron (MLP) is used for qubit readout. Two models are shared, a baseline and a "tiny" version.
Quantization-aware training (QAT) is performed on the tiny model in [HAWQ](https://github.com/Zhen-Dong/HAWQ) and [Brevitas](https://github.com/Xilinx/brevitas).

### Tiny Model Summary 
```
Classifier(
  (classifier): Sequential(
    (0): Linear(in_features=2000, out_features=250, bias=True)
    (1): ReLU(inplace=True)
    (2): BatchNorm1d(250, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Linear(in_features=250, out_features=2, bias=True)
    (4): ReLU(inplace=True)
  )
)
```

## Inference and Synthesis
The target device for the QICK system is the Zynq UltraScale+ [RFSoC ZCU216](https://www.xilinx.com/products/boards-and-kits/zcu216.html) Evaluation Kit. We use Vivado 2020.1 for all synthesis results.

To convert a trained model into FPGA firmware run
```bash 
cd inference
python convert.py -c pytorch/baseline.yml
```

### Quantized Models
Create the environment for QONNX ingestion in hls4ml
```
conda env create -f environment-onnx.yml
conda activate ml4qick-onnx-env
```

To convert a quantized model to FPGA firmware run
```bash
cd inference 
python convert.py -c <framework>/hls_config.yml
```

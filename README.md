# ml-quantum-readout

This project uses hls4ml to create the firmware implementation of machine learning algorithms for single and mult-qubit readout.

## Setup

Dependencies

* Python==3.7
* PyTorch==1.12.0
* keras==2.9.0
* qkeras==0.9.0
* hls4ml==0.6.0

Clone repository

```bash
git clone --recursive https://github.com/jicampos/ml-quantum-readout.git
```

Create conda environment

```bash
conda env create -f environment.yml
conda activate ml4qick-env
```

## Data

Single and mult-qubit data can be found [here](https://urldefense.proofpoint.com/v2/url?u=https-3A__purdue0-2Dmy.sharepoint.com_-3Af-3A_g_personal_oyesilyu-5Fpurdue-5Fedu_EuhbLM-2DwFApNiX9Mh5ZMeIEBG3dGqSIPgwN21j5S30nxvQ-3Fe-3DCDc3Xi&d=DwMFAg&c=gRgGjJ3BkIsb5y6s49QqsA&r=3tXuppM5Ux2UBnxU0DCrdSagIS9IpvGOlIFtsYfyWuc&m=5R-PzD5Udxkr2BBA9AYXREVhYselyKDYk_-1g6QMka_dPV3VTCVJe4id5PFOgpLq&s=fUu9yFLybrPN_AYcDhfBiQoXf5RlOAwbo6DmsD3CiqU&e=).

## Training

A Multi-Layer Perceptron (MLP) is used for qubit readout. A 'baseline' model is established, then is compressed via architecture design and quantization.
Quantization-aware training (QAT) is performed in [HAWQ](https://github.com/Zhen-Dong/HAWQ) and [Qkeras](https://github.com/google/qkeras).

### Model Summary

```text
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 fc1 (Dense)                 (None, 250)               500250    
                                                                 
 relu1 (Activation)          (None, 250)               0         
                                                                 
 batch_normalization_14 (Bat  (None, 250)              1000      
 chNormalization)                                                
                                                                 
 fc2 (Dense)                 (None, 2)                 502       
                                                                 
 relu2 (Activation)          (None, 2)                 0         
                                                                 
=================================================================
Total params: 501,752
Trainable params: 501,252
Non-trainable params: 500
_________________________________________________________________
```

## Inference and Synthesis

The target device for the QICK system is the Zynq UltraScale+ [RFSoC ZCU216](https://www.xilinx.com/products/boards-and-kits/zcu216.html) Evaluation Kit. We use Vivado 2020.1 for all synthesis results.

```bash
cd inference 
python convert.py -c <framework>/<config>.yml
```

## Useful Links

* [ZCU216 Evaluation Board User Guide](https://docs.xilinx.com/v/u/en-US/ug1390-zcu216-eval-bd)
* [ZCU216 PYNQ](https://github.com/sarafs1926/ZCU216-PYNQ)

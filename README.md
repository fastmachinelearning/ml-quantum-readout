# ml-quantum-readout

This project uses hls4ml to create the firmware implementation of machine learning algorithms for single and mult-qubit readout.

## Setup

Clone repository

```bash
git clone https://github.com/jicampos/ml-quantum-readout.git
```

Create conda environment

```bash
conda env create -f environment.yml
conda activate ml4qick-env
```

## Data

Single qubit data can be found [here](https://purdue0-my.sharepoint.com/personal/du245_purdue_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fdu245%5Fpurdue%5Fedu%2FDocuments%2FShared%2FQSC%20ML%20for%20readout%2FFinal%5Fraw%5Fdata%5Ffor%5Fpaper%2Fdata%5F0528%5Fnpy).
Previous versions exist, which can be found in [data dir](data/README.md).

## Training & Notebooks

Several notebooks for training exist, the simpliest are the 'workflow*.ipynb' notebooks. These notebooks start with training in (Q)Keras down to hls4ml IP generation. The scanning notebooks are useful for design space exploration, comparing traditional methods (match filtering and thresholding) with NNs of varying sizes.

### Model Summary

```text
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 fc1 (QDense)                (None, 14)                11214     
                                                                 
 batchnorm1 (QBatchNormaliza  (None, 14)               56        
 tion)                                                           
                                                                 
 fc2 (QDense)                (None, 2)                 30        
                                                                 
=================================================================
Total params: 11,300
Trainable params: 11,272
Non-trainable params: 28
```

## Inference and Synthesis

The target device for the QICK system is the Zynq UltraScale+ [RFSoC ZCU216](https://www.xilinx.com/products/boards-and-kits/zcu216.html) Evaluation Kit. We use Vivado 2020.1 for all synthesis results.
<!-- 
```bash
cd inference 
python convert.py -c <framework>/<config>.yml
``` -->

## Useful Links

* [ZCU216 Evaluation Board User Guide](https://docs.xilinx.com/v/u/en-US/ug1390-zcu216-eval-bd)
* [ZCU216 PYNQ](https://github.com/sarafs1926/ZCU216-PYNQ)

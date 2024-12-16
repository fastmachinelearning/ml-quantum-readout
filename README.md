# ml-quantum-readout

This project uses hls4ml to create the firmware implementation of machine learning algorithms for single qubit readout on the QICK system.

## Data

Data is directly from the ADC units of the ZCU216, prepared into npy files and can be found on Zenodo [here](https://zenodo.org/records/14427490).

## Training & Notebooks

There are several examples for training, the simpliest are the workflow notebooks. These notebooks start with training in (Q)Keras down to hls4ml IP generation. The scanning notebooks are useful for design space exploration, comparing traditional methods (match filtering and thresholding) with NNs of varying sizes.

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

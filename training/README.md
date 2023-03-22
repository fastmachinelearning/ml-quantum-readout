# Training

List of saved checkpoints can be found [here](https://fermicloud-my.sharepoint.com/:f:/g/personal/jcampos_services_fnal_gov/Eneb82SL2s5ItPpzcefppf0B6uPbiELoxlybgFL-i4HU_w?e=og90Lf).

## Model Summary

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

## Start training

Run the script to obtain a trained model with a saved checkpoint.

```bash
python FcNN_SingleQubit_RawData_tinyModel.py
```

## Quantization-aware Training

Switch to `src_keras` directory and run `train.py`. Specify the number of epochs, learning rate, batch size and other hyperparameters through command line arguments. All options are shown with `python train.py --help`.

```bash
cd src_keras
python train.py --quantize
```

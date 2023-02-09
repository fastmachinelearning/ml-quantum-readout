# Training 
List of saved checkpoints can be found [here](https://fermicloud-my.sharepoint.com/:f:/g/personal/jcampos_services_fnal_gov/Eneb82SL2s5ItPpzcefppf0B6uPbiELoxlybgFL-i4HU_w?e=og90Lf). 

## Model Summary (tiny-version)
```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Classifier                               [1, 2]                    --
├─Sequential: 1-1                        [1, 2]                    --
│    └─Linear: 2-1                       [1, 16000]                64,016,000
│    └─ReLU: 2-2                         [1, 16000]                --
│    └─Linear: 2-3                       [1, 16000]                256,016,000
│    └─ReLU: 2-4                         [1, 16000]                --
│    └─Dropout: 2-5                      [1, 16000]                --
│    └─Linear: 2-6                       [1, 2000]                 32,002,000
│    └─ReLU: 2-7                         [1, 2000]                 --
│    └─BatchNorm1d: 2-8                  [1, 2000]                 --
│    └─Dropout: 2-9                      [1, 2000]                 --
│    └─Linear: 2-10                      [1, 2]                    4,002
│    └─ReLU: 2-11                        [1, 2]                    --
==========================================================================================
Total params: 352,038,002
Trainable params: 352,038,002
Non-trainable params: 0
Total mult-adds (M): 352.04
==========================================================================================
```


## Start training
Run the training script to obtain an optimized checkpoint.
```bash
$ python FcNN_SingleQubit_RawData_tinyModel.py
```

## Quantization-aware Training 

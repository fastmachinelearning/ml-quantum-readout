# Data

All datasets can be downloaded from [sharepoint](https://urldefense.proofpoint.com/v2/url?u=https-3A__purdue0-2Dmy.sharepoint.com_-3Af-3A_g_personal_oyesilyu-5Fpurdue-5Fedu_EuhbLM-2DwFApNiX9Mh5ZMeIEBG3dGqSIPgwN21j5S30nxvQ-3Fe-3DCDc3Xi&d=DwMFAg&c=gRgGjJ3BkIsb5y6s49QqsA&r=3tXuppM5Ux2UBnxU0DCrdSagIS9IpvGOlIFtsYfyWuc&m=5R-PzD5Udxkr2BBA9AYXREVhYselyKDYk_-1g6QMka_dPV3VTCVJe4id5PFOgpLq&s=fUu9yFLybrPN_AYcDhfBiQoXf5RlOAwbo6DmsD3CiqU&e=).

## Single Qubit Data

Description: Real raw data. Readout time is `2000ns` with 1 sample taken every `1ns`.

- File : **00002_IQ_plot_raw.h5**
  - Keys:
    - `I_e`, `I_g`, `Q_e`, `Q_g`: I and Q for excited and ground state
    - `adc_e_1`, `adc_e_2`, `adc_g_1`, `adc_g_2`: I and Q for excited and ground state

A train, validation, and test split. You can get a different one with a different seed.

- Files: **X_train_val.npy**, **y_train_val.npy**
  - Train and validation split [0.9]
    - <tt>X shape</tt> : (9000, 2000)
    - <tt>y shape</tt> : (9000, 2)
- Files: **X_test.npy**, **y_test.npy**
  - Test split [0.1]
    - <tt>X shape</tt> : (1000, 2000)
    - <tt>y shape</tt> : (1000, 2)

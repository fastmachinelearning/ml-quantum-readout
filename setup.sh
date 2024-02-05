#!/bin/bash

if conda activate ml4qick-env; then 
    echo "Conda environment activated."
else
    echo "Conda environment ml4qick-env does not exist. Setting up environment..."
    conda env create -f environment.yml
fi

XILINX_DIR="/data/Xilinx/Vivado/2020.1/settings64.sh"
echo "Setting up Xilinx Vivado."

source $XILINX_DIR
export XILINXD_LICENSE_FILE=2100@xilinx-lic
export LM_LICENSE_FILE=2100@xilinx-lic

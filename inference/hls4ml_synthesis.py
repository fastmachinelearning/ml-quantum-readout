import os
import argparse
import numpy as np
import hls4ml

import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.utils.data

from utils.data import test_data
from utils.config import print_dict
from utils.hls import evaluate_hls


csr = range(500, 1500)
sr = len(csr)
hn = sr * 2 * 1


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.linear1 = nn.Linear(sr * 2, int(hn / 8))
        self.relu1 = nn.ReLU()
        self.bn = nn.BatchNorm1d(int(hn / 8), affine=True)

        self.linear2 = nn.Linear(int(hn / 8), 2)
        self.relu2 = nn.ReLU()

    def forward(self, sig):
        x = self.linear1(sig)
        x = self.relu1(x)
        x = self.bn(x)

        x = self.linear2(x)
        x = self.relu2(x)
        return x


def get_static_config():
    reusefactor = args.reuse_factor
    weight_precision = 'ap_fixed<12,2>'
    result_precision = 'ap_fixed<24,10>'
    reduced_result_precision = 'ap_fixed<12,4>'

    hls_config ={}

    hls_config['Model'] = {}
    hls_config['LayerName'] = {}
    hls_config['Model']['ReuseFactor'] = reusefactor
    hls_config['Model']['Precision'] = "ap_fixed<10,10>"
    hls_config['Model']['Strategy'] = 'resource'

    ### layers
    hls_config['LayerName']['bn'] = {}
    hls_config['LayerName']['linear1'] = {}
    hls_config['LayerName']['linear2'] = {}
    hls_config['LayerName']['bn']['Precision'] = {}
    hls_config['LayerName']['linear1']['Precision'] = {}
    hls_config['LayerName']['linear2']['Precision'] = {}

    # weight
    hls_config['LayerName']['bn']['Precision']['scale'] = weight_precision
    hls_config['LayerName']['linear1']['Precision']['weight'] = weight_precision
    hls_config['LayerName']['linear2']['Precision']['weight'] = weight_precision

    # bias
    hls_config['LayerName']['bn']['Precision']['bias'] = weight_precision
    hls_config['LayerName']['linear1']['Precision']['bias'] = weight_precision
    hls_config['LayerName']['linear2']['Precision']['bias'] = weight_precision

    # accum 
    hls_config['LayerName']['bn']['Precision']['accum_t'] = result_precision
    hls_config['LayerName']['linear1']['Precision']['accum_t'] = result_precision
    hls_config['LayerName']['linear2']['Precision']['accum_t'] = result_precision
    hls_config['LayerName']['bn']['accum_t'] = result_precision
    hls_config['LayerName']['linear1']['accum_t'] = result_precision
    hls_config['LayerName']['linear2']['accum_t'] = result_precision

    # result
    hls_config['LayerName']['bn']['Precision']['result'] = reduced_result_precision
    hls_config['LayerName']['linear1']['Precision']['result'] = result_precision
    hls_config['LayerName']['linear2']['Precision']['result'] = reduced_result_precision

    # reusefactor
    hls_config['LayerName']['bn']['ReuseFactor'] = reusefactor
    hls_config['LayerName']['linear1']['ReuseFactor'] = reusefactor
    hls_config['LayerName']['linear2']['ReuseFactor'] = 500

    ## activation
    hls_config['LayerName']['relu1'] = {}
    hls_config['LayerName']['relu2'] = {}
    hls_config['LayerName']['relu1']['Precision'] = {}
    hls_config['LayerName']['relu2']['Precision'] = {}
    hls_config['LayerName']['relu1']['Precision'] = result_precision
    hls_config['LayerName']['relu2']['Precision'] = reduced_result_precision
    hls_config['LayerName']['relu1']['ReuseFactor'] = reusefactor
    hls_config['LayerName']['relu2']['ReuseFactor'] = reusefactor
    return hls_config 


def main(args):

    base_dir = "../hls4ml_prjs"
    prj_dir = os.path.join(base_dir, f"hls4ml_prj_resource_rf{args.reuse_factor}")
    hls_fig = os.path.join(prj_dir, "hls_model.png")

    model = Classifier()
    print('Loading model checkpoint...')
    model.load_state_dict(torch.load("../checkpoints/checkpoint_tiny_affine.pth"))

    hls_config = get_static_config()

    print("------------------------------------------------------")
    # print_dict(hls_config)
    print("------------------------------------------------------")

    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model,
        input_shape=[1, 2000],
        hls_config=hls_config,
        output_dir=prj_dir,
        part="xczu49dr-ffvf1760-2-e",
    )

    # compile and compare
    print(f'Creating hls4ml project directory {prj_dir}')
    hls_model.compile()

    # visualize model
    hls4ml.utils.plot_model(
        hls_model, show_shapes=True, show_precision=True, to_file=hls_fig
    )

    # evaluate hls model
    if args.evaluate:
        hls_acc = evaluate_hls(hls_model, test_data)

        print("------------------------------------------------------")
        print(f"hls4ml fidelity: {hls_acc:.6f}")
        print("------------------------------------------------------")

    if args.build:
        hls_model.build(csim=False)
        hls4ml.report.read_vivado_report(prj_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Options for hls4ml synthesis")
    parser.add_argument("--reuse-factor", type=int, default=1)
    parser.add_argument("-b", "--build", action="store_true")
    parser.add_argument("-e", "--evaluate", action="store_true")
    args = parser.parse_args()

    main(args)

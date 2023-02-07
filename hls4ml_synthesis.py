import os
import argparse
import numpy as np
import hls4ml

import torch
import torch.nn as nn
import torch.utils.data

from utils.data import test_data


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
    precision = 'ap_fixed<24,10>'
    reduced_precision = 'ap_fixed<24,10>'
    reusefactor = args.reuse_factor

    config ={}

    config['Model'] = {}
    config['Model']['ReuseFactor'] = reusefactor
    config['Model']['Precision'] = reduced_precision
    config['Model']['Strategy'] = 'resource'
    ### layers
    config['Model']['bn'] = {}
    config['Model']['linear1'] = {}
    config['Model']['linear2'] = {}
    config['Model']['bn']['Precision'] = {}
    config['Model']['linear1']['Precision'] = {}
    config['Model']['linear2']['Precision'] = {}
    # weight
    config['Model']['bn']['Precision']['scale'] = precision
    config['Model']['linear1']['Precision']['weight'] = reduced_precision
    config['Model']['linear2']['Precision']['weight'] = reduced_precision
    # bias
    config['Model']['bn']['Precision']['bias'] = precision
    config['Model']['linear1']['Precision']['bias'] = reduced_precision
    config['Model']['linear2']['Precision']['bias'] = reduced_precision
    # accum 
    config['Model']['bn']['Precision']['accum_t'] = 'ap_fixed<24, 10>'
    config['Model']['linear1']['Precision']['accum_t'] = 'ap_fixed<24, 10>'
    config['Model']['linear2']['Precision']['accum_t'] = 'ap_fixed<24, 10>'
    config['Model']['bn']['accum_t'] = 'ap_fixed<24, 16>'
    config['Model']['linear1']['accum_t'] = 'ap_fixed<24, 16>'
    config['Model']['linear2']['accum_t'] = 'ap_fixed<24, 16>'
    # result
    config['Model']['bn']['Precision']['result'] = 'ap_fixed<24, 16>'
    config['Model']['linear1']['Precision']['result'] = 'ap_fixed<24, 16>'
    config['Model']['linear2']['Precision']['result'] = 'ap_fixed<24, 16>'
    # reusefactor
    config['Model']['bn']['ReuseFactor'] = reusefactor
    config['Model']['linear1']['ReuseFactor'] = reusefactor
    config['Model']['linear2']['ReuseFactor'] = 500
    ## activation
    config['Model']['relu1'] = {}
    config['Model']['relu2'] = {}
    config['Model']['relu1']['Precision'] = {}
    config['Model']['relu2']['Precision'] = {}
    config['Model']['relu1']['Precision'] = precision
    config['Model']['relu2']['Precision'] = precision
    config['Model']['relu1']['ReuseFactor'] = reusefactor
    config['Model']['relu2']['ReuseFactor'] = reusefactor
    return config 


def main(args):

    base_dir = "hls4ml_prjs"
    prj_dir = os.path.join(base_dir, f"hls4ml_prj_resource_rf{args.reuse_factor}")
    hls_fig = os.path.join(prj_dir, "hls_model.png")

    model = Classifier()
    print('Loading model checkpoint...')
    model.load_state_dict(torch.load("checkpoints/checkpoint_tiny_affine.pth"))

    # config = hls4ml.utils.config_from_onnx_model(model, granularity="model")
    # config["Model"]["ReuseFactor"] = args.reuse_factor
    # config["Model"]["Strategy"] = "resource"
    # config["Model"]["Precision"] = "ap_fixed<26, 16>"
    hls_config = get_static_config()

    print("------------------------------------------------------")
    print(hls_config)
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
        correct = 0
        total = 0

        for idx in range(400):
            data, target = test_data[idx]

            data = np.ascontiguousarray(data.numpy())

            states = hls_model.predict(data)
            target = target.numpy()
            pred = np.argmax(states).astype(np.int32)

            if pred - target == 0:
                correct += 1
            total += 1

        print("------------------------------------------------------")
        print(f"hls4ml fidelity: {correct/total:.2f}")
        print("------------------------------------------------------")

    if args.build:
        hls_model.build(csim=False)
        hls4ml.report.read_vivado_report(prj_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Options for hls4ml synthesis")
    parser.add_argument("--reuse-factor", type=int, default=1)
    parser.add_argument("-p", "--precision", type=str, default="ap_fixed<16,6>")
    parser.add_argument("-b", "--build", action="store_true")
    parser.add_argument("-e", "--evaluate", action="store_true")
    args = parser.parse_args()

    main(args)

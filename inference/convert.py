import os
import argparse
import onnx
import yaml
import hls4ml
import torch

import sys
sys.path.append("..")

from utils.data import test_data
from utils.config import print_dict
from utils.hls import evaluate_hls
from training.qat import TinyClassifier


def open_config(args):
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    return config


def main(args):
    config = open_config(args)

    HLSConfig = config["HLSConfig"]
    XilinxPart = config["Part"]
    IOType = config["IOType"]
    ClockPeriod = config["ClockPeriod"]
    ModelCkp = config["ModelCkp"]
    ModelType = config["ModelType"]
    OutputDir = config["OutputDir"]
    HLSFig = os.path.join(OutputDir, "hls_model.png")

    print("------------------------------------------------------")
    print_dict(config)
    print("------------------------------------------------------")

    if ModelType.lower() == "torch":
        model = TinyClassifier()
        model.load_state_dict(torch.load(ModelCkp))

        hls_model = hls4ml.converters.convert_from_pytorch_model(
            model=model,
            input_shape=[1, 2000],
            hls_config=HLSConfig,
            output_dir=OutputDir,
            part=XilinxPart,
            io_type=IOType,
            clock_period=ClockPeriod,
        )
    elif ModelType.lower() == "onnx":
        model = onnx.load(ModelCkp)
        hls_model = hls4ml.converters.convert_from_onnx_model(
            model=model,
            hls_config=HLSConfig,
            output_dir=OutputDir,
            part=XilinxPart,
            io_type=IOType,
            clock_period=ClockPeriod,
        )

    # compile and compare
    print(f"Creating hls4ml project directory {OutputDir}")
    hls_model.compile()

    # visualize model
    hls4ml.utils.plot_model(
        hls_model, show_shapes=True, show_precision=True, to_file=HLSFig
    )

    # evaluate hls model
    if args.evaluate:
        hls_acc = evaluate_hls(hls_model, test_data)

        print("------------------------------------------------------")
        print(f"hls4ml fidelity: {hls_acc:.6f}")
        print("------------------------------------------------------")

    if args.build:
        BuildOptions = config["BuildOptions"]
        for opt in BuildOptions:
            BuildOptions[opt] = True if BuildOptions[opt] == 1 else False
        hls_model.build(
            reset=BuildOptions["reset"],
            csim=BuildOptions["csim"],
            synth=BuildOptions["synth"],
            cosim=BuildOptions["cosim"],
            validation=BuildOptions["validation"],
            export=BuildOptions["export"],
            vsynth=BuildOptions["vsynth"],
            fifo_opt=BuildOptions["fifo_opt"],
        )
        hls4ml.report.read_vivado_report(OutputDir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Options for hls4ml")
    parser.add_argument("-c", "--config", type=str, default="pytorch/baseline.yml")
    parser.add_argument("-b", "--build", action="store_true")
    parser.add_argument("-e", "--evaluate", action="store_true")
    args = parser.parse_args()

    main(args)

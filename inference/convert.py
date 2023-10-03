import os
import sys
sys.path.append("..")
import sys
import argparse
import onnx
import yaml
import hls4ml
from datetime import datetime
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects

from utils.data import get_dataset
from utils.config import print_dict
from utils.hls import evaluate_hls


def open_config(args):
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    return config


def load_data(data_dir):
    X_test = np.ascontiguousarray(np.load(os.path.join(data_dir, 'X_test.npy')))    
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'), allow_pickle=True)
    return X_test, y_test


def main(args):
    config = open_config(args)

    HLSConfig =      config["HLSConfig"]
    XilinxPart =     config["Part"]
    Board =          config['Board']
    Interface =      config["Interface"]
    Backend =        config["Backend"]
    Driver =         config["Driver"]
    IOType =         config["IOType"]
    ClockPeriod =    config["ClockPeriod"]
    ModelCkp =       config["ModelCkp"]
    ModelType =      config["ModelType"]
    ModelFramework = config["Framework"]
    DataDir =        config['DataDir']
    OutputDir =      config["OutputDir"]
    HLSFig =         os.path.join(OutputDir, "hls_model.png")

    print("------------------------------------------------------")
    print_dict(config)
    print("------------------------------------------------------")

    if ModelFramework.lower() == "torch":
        model.load_state_dict(torch.load(ModelCkp))
        print(model)

        hls_model = hls4ml.converters.convert_from_pytorch_model(
            model=model,
            input_shape=[1, 2000],
            hls_config=HLSConfig,
            output_dir=OutputDir,
            part=XilinxPart,
            io_type=IOType,
            clock_period=ClockPeriod,
        )
    elif ModelFramework.lower() == "hawq":
        model = onnx.load(ModelCkp)
        
        hls_model = hls4ml.converters.convert_from_onnx_model(
            model=model,
            hls_config=HLSConfig,
            output_dir=OutputDir,
            part=XilinxPart,
            io_type=IOType,
            clock_period=ClockPeriod,
        )
    elif ModelFramework.lower() == "keras":
        co = {}
        _add_supported_quantized_objects(co)
        model = load_model(ModelCkp, custom_objects=co, compile=False)

        hls_model = hls4ml.converters.convert_from_keras_model(
            model=model,
            hls_config=HLSConfig,
            output_dir=OutputDir,
            part=XilinxPart,
            io_type=IOType,
            clock_period=ClockPeriod,
            backend=Backend,
            board=Board,
            interface=Interface,
            driver=Driver,
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
        X_test, y_test = load_data(DataDir)
        y_keras = model.predict(X_test)
        y_hls = hls_model.predict(np.ascontiguousarray(X_test)) 
        print(y_hls.shape)
        print(y_test.shape)
        hls_acc = accuracy_score(y_test, np.argmax(y_hls, axis=1)) 
        keras_acc = accuracy_score(y_test, np.argmax(y_keras, axis=1)) 

        np.savetxt(os.path.join(OutputDir, 'y_test.dat'), y_test, fmt='%i')
        np.savetxt(os.path.join(OutputDir, 'y_keras.dat'), y_keras, fmt='%.10f')
        np.savetxt(os.path.join(OutputDir, 'y_hls.dat'), y_hls, fmt='%.10f')
        np.savetxt(os.path.join(OutputDir, 'inputs.dat'), X_test, fmt='%i')
        
        print("------------------------------------------------------")
        print(f"hls4ml fidelity: {hls_acc:.6f}")
        print(f"keras fidelity: {keras_acc:.6f}")
        print("------------------------------------------------------")

        hls_acc = accuracy_score(y_test[:20], np.argmax(y_hls[:20], axis=1)) 
        print("------------------------------------------------------")
        print(f"hls4ml fidelity (first 20): {hls_acc:.6f}")
        print("------------------------------------------------------")

        print('Saving inputs and predictions...')
        np.savetxt(os.path.join(OutputDir, 'inputs.dat'), X_test, fmt='%i')  # (1000, 800)
        # Save predictions (output precision ap_fixed<16,3>)
        np.savetxt(os.path.join(OutputDir, 'y_hls.dat'), y_hls, fmt='%.13f')
        np.savetxt(os.path.join(OutputDir, 'y_keras.dat'), y_keras, fmt='%.13f')

    if args.build:
        start_time = datetime.now()

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

        time_elapsed = datetime.now() - start_time
        print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Options for hls4ml")
    parser.add_argument("-c", "--config", type=str, default="pytorch/baseline.yml")
    parser.add_argument("-b", "--build", action="store_true")
    parser.add_argument("-e", "--evaluate", action="store_true")
    args = parser.parse_args()

    main(args)

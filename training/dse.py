"""
"""
from __future__ import print_function
import os
import argparse
import pickle
from datetime import datetime
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# tensorflow/keras imports 
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy

# qkeras and pruning imports 
from qkeras.qlayers import QDense, QActivation
from qkeras import QBatchNormalization
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.utils import _add_supported_quantized_objects
from tensorflow.keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects

from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning
import tensorflow_model_optimization as tfmot


#########################################################
# Helper functions
#########################################################
CHECKPOINT_DIR = 'checkpoints/'
CHECKPOINT_FILENAME = 'model_best.h5'

def get_checkpoint_filename(experiment_num):
    return os.path.join(CHECKPOINT_DIR, f'{experiment_num}', CHECKPOINT_FILENAME)

def get_callbacks(checkpoint_exp_filename, prune):
    callbacks = [
            ModelCheckpoint(
            checkpoint_exp_filename,
            monitor="val_loss",
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            save_freq="epoch",
        ),
        ReduceLROnPlateau(patience=75, min_delta=1**-6),
    ]
    if prune:
        callbacks.append(pruning_callbacks.UpdatePruningStep())
    return callbacks

def load_checkpoint(filename):
    co = {}
    _add_supported_quantized_objects(co)
    model = load_model(filename, custom_objects=co, compile=False)
    return model

def one_hot_encode(data):
    y_encoded = np.zeros([data.shape[0],2], dtype=np.int32)
    for idx, x in enumerate(data):
        if x == 1:
            y_encoded[idx][1] = 1
        else:
            y_encoded[idx][0] = 1
    return y_encoded

def load_data(data_dir, window_start, window_end, bit_shift):
    print('#########################################################')
    print(f'Loading data from dir: {data_dir}')
    print('\tWindow start:', window_start)
    print('\tWindow end:', window_end)
    print('#########################################################')
    X_train_val = np.load(os.path.join(data_dir, f'X_train_{window_start}_{window_end}.npy'))
    X_test = np.load(os.path.join(data_dir, f'X_test_{window_start}_{window_end}.npy'))    
    y_train_val = np.load(os.path.join(data_dir, f'y_train_{window_start}_{window_end}.npy'))
    y_test = np.load(os.path.join(data_dir, f'y_test_{window_start}_{window_end}.npy'))

    y_train_val = one_hot_encode(y_train_val)
    y_test = one_hot_encode(y_test)

    if bit_shift:
        shift = 3
        print(X_train_val.max())
        X_train_val = np.right_shift(X_train_val.astype(np.int32), shift)
        X_test = np.right_shift(X_test.astype(np.int32), shift)
        print(X_train_val.max())

    return X_train_val, X_test, y_train_val, y_test


#########################################################
# Model
#########################################################
def build_model(model_type, input_shape, quantize, is_pruning):
    model = Sequential()
    # build model 
    if model_type == 'mlp':
        # csr = range(0, 770)
        # sr = len(csr)
        # hn = sr*2 * 1
        sr = int(input_shape/2)
        hn = sr * 2
        if quantize:
            model.add(QDense(int(hn/8), input_shape=(sr*2,), name='fc1', kernel_quantizer=quantized_bits(6,0,alpha=1), bias_quantizer=quantized_bits(6,0,alpha=1),))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(QDense(2, name='fc2', kernel_quantizer=quantized_bits(6,0,alpha=1), bias_quantizer=quantized_bits(6,0,alpha=1)))
            model.add(Activation('relu'))
        else:
            model.add(Dense(int(hn/8), activation='relu', input_shape=(sr*2,)))
            model.add(BatchNormalization())
            model.add(Dense(2, activation='relu'))
    elif model_type == 'single':
        if quantize: 
            model.add(QDense(2, input_shape=(input_shape,), name='fc1', kernel_quantizer=quantized_bits(6,0,alpha=1), bias_quantizer=quantized_bits(6,0,alpha=1),))
            model.add(BatchNormalization())
        else:
            model.add(Dense(2, input_shape=(input_shape,), name='fc1'))
            model.add(BatchNormalization())
    # pruning options 
    if is_pruning == True:
        pruning_params = {"pruning_schedule": pruning_schedule.ConstantSparsity(0.85, begin_step=200, frequency=100)}
        # Define model for pruning
        pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50, final_sparsity=0.80, begin_step=200, end_step=1000)}
        model = prune.prune_low_magnitude(model, **pruning_params)
    return model 

#########################################################
# Training 
#########################################################
def train(model, X_train_val, y_train_val, lr, batch_size, epochs, val_split, callbacks):
    initial_learning_rate = lr
    opt = Adam(learning_rate=initial_learning_rate)
    model.compile(optimizer=opt, loss=CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    history = model.fit(X_train_val, 
              y_train_val, 
              batch_size=batch_size,
              epochs=epochs, 
              validation_split=val_split, 
              shuffle=True, 
              callbacks=callbacks
    )
    return history


#########################################################
# Main
#########################################################
def main(args):
    start_time = datetime.now()

    X_train, X_test, y_train, y_test = load_data(args.data_dir, args.window_start, args.window_end, args.bit_shift)
    model = build_model(args.model_type, X_train.shape[1], args.quantize, args.prune)
    model.build((args.batch_size, X_train.shape[1]))
    print('=============================Model Summary=============================')
    print(model.summary())
    print('=======================================================================')

    checkpoint_exp_filename = get_checkpoint_filename(args.exp)
    callbacks = get_callbacks(checkpoint_exp_filename, args.prune)
    history = train(model, X_train, y_train, args.lr, args.batch_size, args.epochs, args.val_split, callbacks)
    
    if args.prune:
        model = strip_pruning(model)

    print('=============================Model Summary=============================')
    print(model.summary())
    print('=======================================================================')
    
    print(f'Saved checkpoint to: {checkpoint_exp_filename}')
    history_file = checkpoint_exp_filename.replace(".h5", "-history.pkl")
    print(f'Saving history to: {history_file}')
    with open(history_file, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    # get test accuracy
    model = load_checkpoint(checkpoint_exp_filename)
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    print('Keras Test Accuracy: {}'.format(test_acc))

    # get train accuracy
    y_pred = model.predict(X_train)
    train_acc = accuracy_score(np.argmax(y_train, axis=1), np.argmax(y_pred, axis=1))
    print('Keras Train Accuracy: {}'.format(train_acc))

    # model parameters and sparsity 
    model_params_count = model.count_params()
    zero_params_count = 0
    for layer in model.layers:
        if len(layer.get_weights()) > 0:  # Some layers may not have weights
            params = layer.get_weights()
            for param in params:
                zero_params_count += np.sum(param == 0)

    print(f'Number of parameters: {model_params_count}')
    print(f'Number of zero parameters: {zero_params_count}')

    print('=======================================================================')

    # save to file 
    with open('dse-run2.txt', 'a') as file:
        args.quantize = 1 if args.quantize == True else 0
        args.prune = 1 if args.prune == True else 0
        run_results = f'{args.model_type}_{args.window_start}_{args.window_end}, {args.exp}, {args.quantize}, {args.prune}, {args.window_start}, {args.window_end}, {train_acc}, {test_acc}, {model_params_count}, {zero_params_count}\n'
        file.write(run_results)

    time_elapsed = datetime.now() - start_time
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training Options.')
    # data options 
    parser.add_argument('-d', '--data-dir', type=str, default='../../data/new-raw-data-all')
    parser.add_argument('-b', '--batch-size', type=int, default=12800)
    parser.add_argument('--window-start', type=int, default=0)
    parser.add_argument('--window-end', type=int, default=770)
    parser.add_argument('--bit-shift', action='store_true')
    # training options 
    parser.add_argument('-e', '--epochs',type=int, default=50)
    parser.add_argument('-v', '--val-split', type=float, default=0.03)
    parser.add_argument('-lr', type=float, default=1e-2)
    # model options 
    parser.add_argument('--model-type', choices=['mlp', 'single'], help='Choose a model type from the list: mlp, single')
    parser.add_argument('-q', '--quantize', action='store_true')
    parser.add_argument('-p', '--prune', action='store_true')
    # saving options 
    parser.add_argument('--exp', type=int, default=0, help='Experiment run number. ')
    args = parser.parse_args()

    main(args)

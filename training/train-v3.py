import os
import argparse
from datetime import datetime
import numpy as np 
import pickle
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy

from qkeras.qlayers import QDense, QActivation
from qkeras import QBatchNormalization
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.utils import _add_supported_quantized_objects
from tensorflow.keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects

from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning
import tensorflow_model_optimization as tfmot


checkpoint_filename = ''

def create_filename(args):
    if args.quantize or args.prune:
        return './qkeras_v3_filter_pnq.h5'
    return './qkeras_v3_filter.h5'

def get_callbacks(args):
    global checkpoint_filename
    checkpoint_filename = create_filename(args)
    callbacks = [
        #     ModelCheckpoint(
        #     checkpoint_filename,
        #     monitor="val_loss",
        #     verbose=0,
        #     save_best_only=True,
        #     save_weights_only=False,
        #     save_freq="epoch",
        # ),
        ReduceLROnPlateau(patience=75, min_delta=1**-6),
    ]
    if args.prune:
        callbacks.append(pruning_callbacks.UpdatePruningStep())
    return callbacks


def one_hot_encode(data):
    y_encoded = np.zeros([data.shape[0],2], dtype=np.int32)
    for idx, x in enumerate(data):
        if x == 1:
            y_encoded[idx][1] = 1
        else:
            y_encoded[idx][0] = 1
    return y_encoded


def load_data(args):
    print(args.data_dir)
    X_train_val = np.load(os.path.join(args.data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(args.data_dir, 'X_test.npy'))    
    y_train_val = np.load(os.path.join(args.data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(args.data_dir, 'y_test.npy'), allow_pickle=True)

    y_train_val = one_hot_encode(y_train_val)
    y_test = one_hot_encode(y_test)

    args.input_shape = X_train_val.shape[1]

    if args.bit_shift:
        shift = 3
        
        print(X_train_val.max())
        
        X_train_val = np.right_shift(X_train_val.astype(np.int32), shift)
        X_test = np.right_shift(X_test.astype(np.int32), shift)

        print(X_train_val.max())

    return X_train_val, X_test, y_train_val, y_test


def load_checkpoint(args, filename):
    co = {}
    _add_supported_quantized_objects(co)
    model = load_model(filename, custom_objects=co, compile=False)
    return model


def get_keras_model():
    model = Sequential()
    model.add(Dense(2, input_shape=(200,), name='fc1'))
    model.add(BatchNormalization())
    return model


def get_qkeras_model(args):
    model = Sequential()
    model.add(QDense(2, input_shape=(200,), name='fc1', kernel_quantizer=quantized_bits(6,0,alpha=1), bias_quantizer=quantized_bits(6,0,alpha=1),))
    model.add(BatchNormalization())
    if args.prune:
        pruning_params = {"pruning_schedule": pruning_schedule.ConstantSparsity(0.70, begin_step=200, frequency=100)}
        pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50, final_sparsity=0.80, begin_step=200, end_step=1000)}
        model = prune.prune_low_magnitude(model, **pruning_params)
    return model


def train(model, X_train_val, y_train_val, args):
    initial_learning_rate = args.lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)
    opt = Adam(learning_rate=initial_learning_rate)
    model.compile(optimizer=opt, loss=CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    history = model.fit(X_train_val, 
              y_train_val, 
              batch_size=args.batch_size,
              epochs=args.epochs, 
              validation_split=args.val_split, 
              shuffle=True, 
              callbacks=get_callbacks(args)
    )
    return history


def main(args):
    start_time = datetime.now()

    X_train, X_test, y_train, y_test = load_data(args)
    model = get_qkeras_model(args) if args.quantize else get_keras_model()

    history = train(model, X_train, y_train, args)
    if args.prune:
        model = strip_pruning(model)
    model.save(create_filename(args))

    print('=======================================================================')
    print(f'Number of training samples: {len(X_train)}')
    print(f'Number of testing samples: {len(X_test)}')
    print('=============================Model Summary=============================')
    print(model.summary())
    print('=======================================================================')
    
    print(f'Saved checkpoint to: {checkpoint_filename}')
    history_file = checkpoint_filename.replace(".h5", "-history.pkl")
    print(f'Saving history to: {history_file}')
    with open(history_file, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    model = load_checkpoint(args, checkpoint_filename)
    y_pred = model.predict(X_test)
    print("Keras  Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))))

    print(f'Number of layers: {len(model.layers)}')
    w = model.layers[0].weights[0].numpy()
    h, b = np.histogram(w, bins=100)
    plt.figure(figsize=(7, 7))
    plt.bar(b[:-1], h, width=b[1] - b[0])
    plt.semilogy()
    plt.savefig('model-v3-filter-pruned.png')
    print('% of zeros = {}'.format(np.sum(w == 0) / np.size(w)))
    w = model.layers[1].weights[0].numpy()
    print('% of zeros = {}'.format(np.sum(w == 0) / np.size(w)))

    print('=======================================================================')
    time_elapsed = datetime.now() - start_time
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training Options.')
    parser.add_argument('-d', '--data-dir', type=str, default='../../data/new-raw-data')
    parser.add_argument('-b', '--batch-size', type=int, default=12800)
    parser.add_argument('-e', '--epochs',type=int, default=25)
    parser.add_argument('-v', '--val-split', type=float, default=0.03)
    parser.add_argument('--bit-shift', action='store_true')
    parser.add_argument('-lr', type=float, default=1e-2)
    parser.add_argument('-q', '--quantize', action='store_true')
    parser.add_argument('-p', '--prune', action='store_true')
    args = parser.parse_args()

    main(args)

# python train-dummy.py --quantize

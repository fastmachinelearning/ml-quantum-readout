import os
import argparse
from datetime import datetime
import numpy as np 
from sklearn.metrics import accuracy_score

import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy

from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.utils import _add_supported_quantized_objects


def create_filename(args):
    if args.quantize:
        return f'../../checkpoints/qkeras/i500_w1000_s5/qkeras_best{args.batch_size}_e{args.epochs}_val{args.val_split}_lr{args.lr}.h5'
    return f'../../checkpoints/keras/keras_best_bs{args.batch_size}_e{args.epochs}_val{args.val_split}_lr{args.lr}.h5'


def get_callbacks(args):
    return [
            ModelCheckpoint(
            create_filename(args),
            monitor="val_loss",
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            save_freq="epoch",
        ),
        ReduceLROnPlateau(patience=75, min_delta=1**-6)
    ]


def one_hot_encode(data):
    y_encoded = np.zeros([data.shape[0],2], dtype=np.int32)
    for idx, x in enumerate(data):
        if x == 1:
            y_encoded[idx][1] = 1
        else:
            y_encoded[idx][0] = 1
    return y_encoded


def load_data(args):
    X_train_val = np.load(os.path.join(args.data_dir, 'X_train_val.npy'))
    X_test = np.load(os.path.join(args.data_dir, 'X_test.npy'))    
    y_train_val = np.load(os.path.join(args.data_dir, 'y_train_val.npy'))
    y_test = np.load(os.path.join(args.data_dir, 'y_test.npy'), allow_pickle=True)

    y_train_val = one_hot_encode(y_train_val)
    y_test = one_hot_encode(y_test)

    return X_train_val, X_test, y_train_val, y_test


def get_keras_model():
    model = Sequential()
    model.add(Dense(250, input_shape=(2000,), name='fc1',))
    model.add(Activation(activation='relu', name='relu1'))
    model.add(BatchNormalization())
    model.add(Dense(2, name='fc2',))
    model.add(Activation(activation='relu', name='relu2'))
    return model


def get_qkeras_model():
    model = Sequential()
    model.add(QDense(50, input_shape=(400,), name='fc1',
                        kernel_quantizer=quantized_bits(2,0,alpha=1), bias_quantizer=quantized_bits(2,0,alpha=1),))
    model.add(QActivation(activation=quantized_relu(6,3), name='relu1'))
    model.add(BatchNormalization())
    model.add(QDense(2, name='fc2',
                        kernel_quantizer=quantized_bits(2,0,alpha=1), bias_quantizer=quantized_bits(2,0,alpha=1),))
    model.add(QActivation(activation=quantized_relu(6,3), name='relu2'))
    return model


def train(model, X_train_val, y_train_val, args):
    adam = Adam(learning_rate=args.lr)
    model.compile(optimizer=adam, loss=CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.fit(X_train_val, 
              y_train_val, 
              batch_size=args.batch_size,
              epochs=args.epochs, 
              validation_split=args.val_split, 
              shuffle=True, 
              callbacks=get_callbacks(args)
    )


def main(args):
    start_time = datetime.now()

    X_train_val, X_test, y_train_val, y_test = load_data(args)
    model = get_qkeras_model() if args.quantize else get_keras_model()

    train(model, X_train_val, y_train_val, args)

    y_pred = model.predict(X_test)
    print("Keras  Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))))

    time_elapsed = datetime.now() - start_time
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training Options.')
    parser.add_argument('-d', '--data-dir', type=str, default='../../data')
    parser.add_argument('-b', '--batch-size', type=int, default=1024)
    parser.add_argument('-e', '--epochs',type=int, default=100)
    parser.add_argument('-v', '--val-split', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-2)
    parser.add_argument('-q', '--quantize', action='store_true')
    args = parser.parse_args()

    main(args)

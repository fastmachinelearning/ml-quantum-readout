import os
import argparse
from datetime import datetime
import numpy as np 
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import regularizers

from qkeras.qlayers import QDense, QActivation
from qkeras.qnormalization import QBatchNormalization
from qkeras.quantizers import quantized_bits, quantized_relu
from tensorflow.keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects


class Model(tf.keras.Model):
    def __init__(self, args) -> None:
        super().__init__()
        # model#1 parameters
        self.model_1_output_shape = args.input_model_output
        self.nodes_fc1 = args.input_model_hidden
        self.num_models = args.num_models
        self.model_1_input_shape = int(args.input_shape / args.num_models)
        # model#2 parameters
        self.model2_input_shape = args.num_models * self.model_1_output_shape
        # if True, reuse use model#1 for all input slices 
        self.reuse = args.reuse
        self.quantize = args.quantize
        self.models = self.get_models(args)

    def get_model_1(self, input_shape):
        if self.quantize:
            model = keras.Sequential()
            model.add(QDense(input_shape, name='layer1', kernel_quantizer=quantized_bits(16,5,alpha=1), bias_quantizer=quantized_bits(16,5,alpha=1),
                             kernel_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-2)))
            model.add(BatchNormalization(name='batchnorm1'),)
            model.add(QActivation(activation=quantized_relu(16,6), name='relu1'))
            model.add(QDense(self.model_1_output_shape, name="layer3", kernel_quantizer=quantized_bits(16,5,alpha=1), bias_quantizer=quantized_bits(16,5,alpha=1),
                             kernel_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-2)))
            return model
        model = keras.Sequential(
            [
                layers.Dense(input_shape, activation="relu", name="layer1", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)),
                layers.Dropout(0.3),
                # layers.Dense(self.nodes_fc1, name="layer2", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)),
                layers.Dense(self.model_1_output_shape, name="layer3", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)),
            ]
        )
        return model

    def get_model_2(self, input_shape):
        if self.quantize:
            model = keras.Sequential()
            model.add(QDense(input_shape, name='layer1', kernel_quantizer=quantized_bits(16,5,alpha=1)))
            model.add(QActivation(activation=quantized_relu(16,6), name='relu1'))
            model.add(QDense(2, name='layer2', kernel_quantizer=quantized_bits(16,5,alpha=1)))
            model.add(Activation(activation='softmax', name='softmax'))
            return model    
        model = keras.Sequential(
            [
                layers.Dense(input_shape, activation="relu", name="layer1"),
                layers.Dense(int(input_shape/2), activation="relu", name="layer2"),
                layers.Dense(2, activation="softmax", name="softmax"),
            ]
        )
        return model

    def get_models(self, args):

        models = list()
        for idx in range(args.num_models):
            if idx>0 and self.reuse:
                models.append(models[idx-1])
            else:
                models.append(self.get_model_1(self.model_1_input_shape))

        models.append(self.get_model_2(self.model2_input_shape))
        return models

    def call(self, x):
        model1_outputs = list()

        for idx in range(self.num_models):
            output = self.models[idx](x[:, idx*self.model_1_input_shape:idx*self.model_1_input_shape+self.model_1_input_shape])
            model1_outputs.append(output)
        
        t = tf.stack(model1_outputs, axis=1)
        t = tf.reshape(t, [-1,self.model2_input_shape])

        outputs = self.models[-1](t)
        return outputs

    def predict(self, x):
        model1_outputs = list()

        for idx in range(self.num_models):
            output = self.models[idx](x[:, idx*self.model_1_input_shape:idx*self.model_1_input_shape+self.model_1_input_shape])
            model1_outputs.append(output)
        
        t = tf.stack(model1_outputs, axis=1)
        t = tf.reshape(t, [-1,self.model2_input_shape])

        outputs = self.models[-1](t)
        return outputs

    def load_checkpoint(self, model1_filename, model2_filename):
        """Only works when Model#1 is reused for all inputs"""
        
        co = {} 
        _add_supported_quantized_objects(co)
        
        self.models[0] = load_model(model1_filename, custom_objects=co, compile=False)
        self.models[0].build(input_shape=[1,self.model_1_input_shape])
        # make pointers to the first model 
        for idx in range(1, len(self.models)-1):
            self.models[idx] = self.models[idx-1]
        self.models[-1] = load_model(model2_filename, custom_objects=co, compile=False)
        self.models[-1].build(input_shape=[1,self.model2_input_shape])


def get_callbacks(args):
    return [
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
    print(args.data_dir)
    X_train_val = np.load(os.path.join(args.data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(args.data_dir, 'X_test.npy'))    
    y_train_val = np.load(os.path.join(args.data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(args.data_dir, 'y_test.npy'), allow_pickle=True)

    y_train_val = one_hot_encode(y_train_val)
    y_test = one_hot_encode(y_test)

    args.input_shape = X_train_val.shape[1]
    return X_train_val, X_test, y_train_val, y_test


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

    model = Model(args)

    if args.train:
        model.build(input_shape=(1,1540))
    #     model.build(input_shape=(1,770))
    #     print('-----------------------------------------------------')
    #     print(f'Input Length: {args.input_shape}')
    #     print(f'Number Models: {len(model.models)-1}')
    #     print('-----------------------------------------------------')
    #     print('Input Model')
    #     print(model.models[0].summary())
    #     print('\n-----------------------------------------------------')
    #     print('Output Model')
    #     print(model.models[-1].summary())

        train(model, X_train_val, y_train_val, args)
    
    if args.ckp:
        print(f'Saving Model#1 as {args.ckp_file_m1}...')
        model.models[0].save(args.ckp_file_m1)
        print(f'Saving Model#2 as {args.ckp_file_m2}...')
        model.models[-1].save(args.ckp_file_m2)

    if args.train:
        print('-----------------------------------------------------')
        print(f'Input Length: {args.input_shape}')
        print(f'Number Models: {len(model.models)-1}')
        print('-----------------------------------------------------')
        print('Input Model')
        print(model.models[0].summary())
        print('\n-----------------------------------------------------')
        print('Output Model')
        print(model.models[-1].summary())

    # load checkpoint and validate model accuracy is the same
    if args.ckp: 
        model.load_checkpoint(args.ckp_file_m1, args.ckp_file_m2)
    # model.load_checkpoint('model_1_b16_i6_ex2.h5', 'model_2_b16_i6_ex2.h5')
    y_pred = model.predict(X_test)
    model_acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

    cce = tf.keras.losses.CategoricalCrossentropy()
    model_loss = cce(y_test, y_pred).numpy()
    
    print("Model Test Accuracy [after loading ckp]: {}".format(model_acc))
    print("Model Test Loss [after loading ckp]: {}".format(model_loss))
    print('-----------------------------------------------------')

    # logging test accuracy, loss and parameters
    if args.log:
        f = open("grid-search-modelv3.txt", "a")
        f.write(f"{model_acc}, {model_loss} - {args}\n")
        f.close()

    time_elapsed = datetime.now() - start_time
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training Options.')
    # model arch parameters 
    parser.add_argument('--num-models', type=int, default=5)
    parser.add_argument('--input-model-hidden', type=int, default=100)
    parser.add_argument('--input-model-output', type=int, default=3)
    parser.add_argument('-q', '--quantize', action='store_true')
    parser.add_argument('-r', '--reuse', action='store_true')
    # training parameters
    parser.add_argument('--train', action='store_true')
    parser.add_argument('-d', '--data-dir', type=str, default='../../data/new-raw-data')
    parser.add_argument('-b', '--batch-size', type=int, default=12800)
    parser.add_argument('-e', '--epochs',type=int, default=250)
    parser.add_argument('-v', '--val-split', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-3)
    # logging 
    parser.add_argument('--ckp', action='store_true')
    parser.add_argument('--ckp-file-m1', type=str, default='modelv3_1_all.h5')
    parser.add_argument('--ckp-file-m2', type=str, default='modelv3_2_all.h5')
    parser.add_argument('--log', action='store_true', default=False)
    parser.add_argument('--log-file', type=str, default='grid-search-modelv3.txt')
    args = parser.parse_args()

    main(args)

# python ensemble.py -d ../../data/all/ -q

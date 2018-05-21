import librosa
import numpy as np
import scipy
import shutil
import pandas as pd
import math
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import losses, models, optimizers
from keras.utils import np_utils
from keras.activations import relu, softmax
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                          ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras.layers import (Convolution1D, Dense, Dropout, GlobalAveragePooling1D,
                       GlobalMaxPool1D, Input, MaxPool1D, concatenate)
from keras.utils import Sequence, to_categorical,multi_gpu_model
from keras.models import load_model
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

class Config(object):
    def __init__(self,
                 sampling_rate=16000, audio_duration=2, n_classes=41,
                 use_mfcc=False, n_folds=10, learning_rate=0.0001,
                 max_epochs=50, n_mfcc=20, gpus=4):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.gpus = gpus

        self.audio_length = self.sampling_rate * self.audio_duration
        if self.use_mfcc:
            self.dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length/512)), 1)
        else:
            self.dim = (self.audio_length, 1)


class DataGenerator(Sequence):
    def __init__(self, config, data_dir, list_IDs, labels=None,
                 batch_size=64, preprocessing_fn=lambda x: x):
        self.config = config
        self.data_dir = data_dir
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.preprocessing_fn = preprocessing_fn
        self.on_epoch_end()
        self.dim = self.config.dim

    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        return self.__data_generation(list_IDs_temp)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))

    def __data_generation(self, list_IDs_temp):
        cur_batch_size = len(list_IDs_temp)
        X = np.empty((cur_batch_size, self.dim))

        input_length = self.config.audio_length
        for i, ID in enumerate(list_IDs_temp):
            file_path = self.data_dir + ID

            # Read and Resample the audio
            data, _ = librosa.core.load(file_path, sr=self.config.sampling_rate,
                                        res_type='kaiser_fast')

            # Random offset / Padding
            if len(data) > input_length:
                max_offset = len(data) - input_length
                offset = np.random.randint(max_offset)
                data = data[offset:(input_length + offset)]
            else:
                if input_length > len(data):
                    max_offset = input_length - len(data)
                    offset = np.random.randint(max_offset)
                else:
                    offset = 0
                data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

            # Normalization + Other Preprocessing
            if self.config.use_mfcc:
                data = librosa.feature.mfcc(data, sr=self.config.sampling_rate,
                                            n_mfcc=self.config.n_mfcc)
                data = np.expand_dims(data, axis=-1)
            else:
                data = self.preprocessing_fn(data)[:, np.newaxis]
            X[i,] = data

        if self.labels is not None:
            y = np.empty(cur_batch_size, dtype=int)
            for i, ID in enumerate(list_IDs_temp):
                y[i] = self.labels[ID]
            return X, to_categorical(y, num_classes=self.config.n_classes)
        else:
            return X

def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+1e-6)
    return data-0.5


def get_1d_dummy_model(config):
    nclass = config.n_classes
    input_length = config.audio_length

    inp = Input(shape=(input_length, 1))
    x = GlobalMaxPool1D()(inp)
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.learning_rate)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model


def get_1d_conv_model(config):
    nclass = config.n_classes
    input_length = 600000

    inp = Input(shape=(input_length, 1))
    x = Convolution1D(16, 9, activation=relu, padding="valid")(inp)
    x = Convolution1D(16, 9, activation=relu, padding="valid")(x)
    x = MaxPool1D(16)(x)
    x = Dropout(rate=0.1)(x)

    x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    x = MaxPool1D(4)(x)
    x = Dropout(rate=0.1)(x)

    x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    x = MaxPool1D(4)(x)
    x = Dropout(rate=0.1)(x)

    x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
    x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(rate=0.2)(x)

    x = Dense(1028, activation=relu)(x)
    x = Dense(128, activation=relu)(x)
    out = Dense(nclass, activation=softmax)(x)


    model = models.Model(inputs=inp, outputs=out)

    model = multi_gpu_model(model, gpus=config.gpus)
    opt = optimizers.Adam(config.learning_rate)
    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model

def generator(x,y,batch_size):
    ylen = len(y)
    loopcount = ylen // batch_size
    while (True):
        i = np.random.randint(0, loopcount)
        yield x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size]

if __name__ == '__main__':

    COMPLETE_RUN = True
    config = Config(sampling_rate=16000, audio_duration=2, n_folds=10, learning_rate=0.001)

    if not COMPLETE_RUN:
        config = Config(sampling_rate=100, audio_duration=1, n_folds=2, max_epochs=1)

    X = np.load('train.npy')
    y = np.load('label.npy')
    X = np.expand_dims(X, axis=2)

    model = get_1d_conv_model(config)
    # X = X[:2000]
    # y=y[:2000]
    y = np_utils.to_categorical(y, config.n_classes)
    # cfg = tf.ConfigProto()
    # cfg.gpu_options.per_process_gpu_memory_fraction = 0.3
    # set_session(tf.Session(config=cfg))

    # model.fit_generator(generator(X,y,batch_size=32),steps_per_epoch=len(X)//32,epochs=10)
    model.fit(X,y,batch_size=32,epochs=10,shuffle=True)
    loss_and_metrics = model.evaluate(X,y,batch_size=32)
    model.save('k2.h5')

    model = load_model('k2.h5')
    test = np.load('eval.npy')
    test = np.expand_dims(test, axis=2)
    test_label = np.load('eval_label.npy')
    # test = X[:1000]
    # test_label = y[:1000]
    pred = model.predict(test)
    pred_idx = np.argmax(pred,axis=1)

    train = pd.read_csv('./train.csv')
    train_names,train_labels = train['fname'],train['label']
    labels = train_labels.values  # labels:ndarray
    LABELS = list(np.unique(labels))
    label_idx = {i:label for i, label in enumerate(LABELS)}
    pred_name=[]
    acc = 0
    for i,n in enumerate(pred_idx):
        if pred_idx[i] == test_label[i]:
            acc +=1
        pred_name.append([train_names[i+3000],label_idx[n]])
    print(pred_name)
    print(acc/1000.0)
    # df = pd.DataFrame(pred_name,columns=['fname','classes'],index=None)
    # print(df)
    # df.to_csv('predict.csv',index=None)



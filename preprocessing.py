# !/usr/bin/python
# -*- coding: UTF-8 -*-

import wave
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn import preprocessing

# i=1
# filepath = "./audio_train/"  # 添加路径
# filename = os.listdir(filepath)  # 得到文件夹下的所有文件名称
# print(filepath + filename[i])
# f = wave.open(filepath + filename[i], 'rb')
# params = f.getparams()
# nchannels, sampwidth, framerate, nframes = params[:4]
# strData = f.readframes(nframes)  # 读取音频，字符串格式
# waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
# # waveData = waveData * 1.0 / (max(abs(waveData)))  # wave幅值归一化
# # plot the wave
# # time = np.arange(0, nframes) * (1.0 / framerate)
# x = np.arange(0,nframes)
# plt.plot(x, waveData)
# plt.xlabel("nframes")
# plt.ylabel("Amplitude")
# plt.title("Single channel wavedata-filename[%d]" % i)
# plt.grid('on')  # 标尺，on：有，off:无。
# plt.savefig(filename[i].split('.')[0])



def padding(data,input_length):
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
    return data

train_path = './audio_train/'
test_path = './audio_test/'
audio_train_files = os.listdir('./audio_train')
audio_test_files = os.listdir('./audio_test')

train = pd.read_csv('./train.csv')
submission = pd.read_csv('./sample_submission.csv')

input_length = 600000

data=[]
train_label = []
# 读取训练数据
train_names,train_labels = train['fname'],train['label']
print(train_labels)
labels = train_labels.values  # labels:ndarray
LABELS = list(np.unique(labels))
label_idx = {label: i for i, label in enumerate(LABELS)}
print(label_idx)
max_len = 0

for i in range(3000):
    # print(train_names[i],train_labels[i])
    f = wave.open(train_path + train_names[i], 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)  # 读取音频，字符串格式
    waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
    if len(waveData) > max_len:
        max_len = len(waveData)
    waveData = preprocessing.scale(waveData) # wave幅值归一化
    waveData = padding(waveData,input_length=input_length)
    data.append(waveData)
    label = label_idx[labels[i]]
    train_label.append(label)
print(len(data))
print(train_label)
# 1.将较长数据切割成几份等长度数据，扩充训练数据

# 2.将所有数据padding成一样长度

np.save('train',data)
np.save('label',train_label)



import numpy as np
import pandas as pd

train = pd.read_csv('./train.csv')
submission = pd.read_csv('./sample_submission.csv')

input_length = 600000

data=[]
train_label = []

train_names,train_labels = train['fname'],train['label']
print(train_labels)
labels = train_labels.values  # labels:ndarray
LABELS = list(np.unique(labels))
label_idx = {label: i for i, label in enumerate(LABELS)}
print(label_idx)
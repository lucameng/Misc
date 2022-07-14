from tensorflow import keras
from keras import layers
from keras.layers.recurrent import LSTM
from tensorflow_docs.vis import embed
from keras.callbacks import ModelCheckpoint
from imutils import paths
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os
import pydotplus

from keras import layers
#from keras.layers import Dense, Flatten, Dropout, GRU, ZeroPadding3D
from keras.layers.recurrent import LSTM




MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2000



frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
print(type(frame_features_input))



frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

# x = LSTM(2048, return_sequences=True, dropout=0.5)(frame_features_input)

# y = LSTM(2048, return_sequences=True, dropout=0.5)(
#     frame_features_input, mask=mask_input
# )


# print(type(x))
# print(type(y))


def training_vis(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']

    # make a figure
    fig = plt.figure(figsize=(8,4))
    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss, label='train_loss')
    ax1.plot(val_loss, label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Training and Validation Data')
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(acc, label='train_acc')
    ax2.plot(val_acc, label='val_acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy  on Training and Validation Data')
    ax2.legend()
    print("plotting history")
    plt.tight_layout()


training_vis()
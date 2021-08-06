from keras.layers import Dropout
from keras.models import Sequential
from keras.layers.convolutional import MaxPooling3D, UpSampling3D, Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
import numpy as np
import pylab as plt

seq = Sequential()
# -- input
seq.add(ConvLSTM2D(filters=64, kernel_size=(5, 5),
                   input_shape=(5, 256, 256, 3),
                   padding='same', return_sequences=True))
seq.add(Dropout(0.5))
# -- 1st C-LSTM
seq.add(ConvLSTM2D(filters=64, kernel_size=(5, 5),
                   padding='same', return_sequences=True))
# -- 2nd C-LSTM
seq.add(ConvLSTM2D(filters=64, kernel_size=(5, 5),
                   padding='same', return_sequences=True))
# -- 1st max pooling
seq.add(MaxPooling3D(pool_size=(1, 2, 2)))
seq.add(Dropout(0.5))

# -- 3rd C-LSTM
seq.add(ConvLSTM2D(filters=64, kernel_size=(5, 5),
                   padding='same', return_sequences=True))
# -- 4th C-LSTM
seq.add(ConvLSTM2D(filters=64, kernel_size=(5, 5),
                   padding='same', return_sequences=True))
# -- 2*2 devolution
seq.add(UpSampling3D(size=(1, 2, 2)))
seq.add(Dropout(0.5))
# -- 3*3 convolution
seq.add(Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same'))
# -- 1*1 convolution
seq.add(Conv3D(filters=2, kernel_size=(1, 1, 1),
               activation='sigmoid',
               padding='same', data_format='channels_last'))
print(seq.summary())
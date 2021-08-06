import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Data Construction


# Model Construction
# -- input
inputs = layers.Input(shape=(None, 256, 256, 1))
# -- dropout 0.5
x = layers.Dropout(0.5)(inputs)
# -- 2 layers of BDC_LSTM
x = layers.ConvLSTM2D(filters=64, kernel_size=(5, 5), padding='same', data_format='channels_last',
                      return_sequences=True, activation="relu", go_backwards=True)(x)
x = layers.ConvLSTM2D(filters=64, kernel_size=(5, 5), padding='same', data_format='channels_last',
                      return_sequences=True, activation="relu", go_backwards=True)(x)
# -- max pooling
x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
# -- dropout 0.5
x = layers.Dropout(0.5)(x)
# -- 2 layers of BDC_LSTM
x = layers.ConvLSTM2D(filters=64, kernel_size=(5, 5), padding='same', data_format='channels_last',
                      return_sequences=True, activation="relu", go_backwards=True)(x)
x = layers.ConvLSTM2D(filters=64, kernel_size=(5, 5), padding='same', data_format='channels_last',
                      return_sequences=False, activation="relu", go_backwards=True)(x)
# -- upsampling
x = layers.UpSampling2D(size=(2, 2))(x)
# -- dropout 0.5
x = layers.Dropout(0.5)(x)
# -- 3*3 conv
x = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(x)
# -- 1*1 conv
outputs = layers.Conv2D(filters=2, kernel_size=(1, 1), activation='sigmoid', padding="same")(x)
# ------- model -------
model = keras.models.Model(inputs, outputs)
model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam())


# Model Training

# Model Prediction

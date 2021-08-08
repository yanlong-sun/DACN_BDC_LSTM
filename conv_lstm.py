import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Load dataset
path = r'../Dataset/training_data/training_data_bmp/masks_npy_in10/'
npy_list = sorted(os.listdir(path))
npy_nums = len(npy_list)
training_data = np.empty([1000, 10, 256, 256, 1], dtype=int)
valid_data = np.empty([100, 10, 256, 256, 1], dtype=int)
for i in range(npy_nums):
    npy_name = path + npy_list[i]
    loaded_npy = np.load(npy_name)
    loaded_npy = np.expand_dims(loaded_npy, axis=-1)
    if i < 1000:
        training_data[i] = loaded_npy
        continue
    elif 1000 <= i < 1100:
        training_data[i-1000] = loaded_npy
        continue
    break


def create_shifted_frames(data):
    x = data[:, 0:data.shape[1] - 1, :, :]
    y = data[:, 1:data.shape[1], :, :]
    return x, y


x_train, y_train = create_shifted_frames(training_data)
x_val, y_val = create_shifted_frames(valid_data)
print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))


# Data Visualization
# Construct a figure on which we will visualize the images.
fig, axes = plt.subplots(3, 3, figsize=(10, 8))
# Plot each of the sequential images for one random data example.
data_choice = np.random.choice(range(len(training_data)), size=1)[0]
for idx, ax in enumerate(axes.flat):
    img = np.squeeze(training_data[data_choice][idx])
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Frame {idx + 1}")
    ax.axis("off")

# Print information and display the figure.
print(f"Displaying frames for example {data_choice}.")
plt.show()

# Model Construction
# Construct the input layer with no definite frame size.
inp = layers.Input(shape=(None, *x_train.shape[2:]))

# We will construct 3 `ConvLSTM2D` layers with batch normalization,
# followed by a `Conv3D` layer for the spatiotemporal outputs.
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(5, 5),
    padding="same",
    return_sequences=True,
    activation="relu",
)(inp)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(3, 3),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(1, 1),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = layers.Conv3D(
    filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
)(x)

# Next, we will build the complete model and compile it.
model = keras.models.Model(inp, x)
model.compile(
    loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),
)

# Define some callbacks to improve training.
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

# Define modifiable training hyperparameters.
epochs = 20
batch_size = 5

# Fit the model to the training data.
model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, reduce_lr],
)
model.save("convLstmModel")
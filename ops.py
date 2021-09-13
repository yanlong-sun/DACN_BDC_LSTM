import tensorflow as tf


def convlstm2d(inputs, filters, kernel_size, stride=1, padding='SAME', data_format='channels_last',
               go_backwards=False, return_sequences=True):
    outputs = tf.keras.layers.ConvLSTM2D(filters, kernel_size, stride, activation='relu', data_format=data_format, padding=padding,
                                         go_backwards=go_backwards, return_sequences=return_sequences)(inputs)
    return outputs


def max_pool_3d(inputs, pool_size=(1, 2, 2), padding='same', strides=None, data_format='channels_last'):
    outputs = tf.keras.layers.MaxPool3D(pool_size, padding=padding, strides=strides, data_format=data_format)(inputs)

    return outputs


def upsamping2d(inputs, size=(2, 2), interpolation='bilinear'):
    outputs = tf.keras.layers.UpSampling2D(size=size, interpolation=interpolation)(inputs)
    return outputs


def conv2d(inputs, filters, kernel_size=3, padding='SAME', data_format='channels_last'):
    outputs = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                     padding=padding, data_format=data_format)(inputs)
    return outputs






"""
def conv2d(inputs, rate_field, num_outputs, kernel_size, scope, stride=1, rate=1,
           is_train=True, bias=True, norm=True, activation=True, d_format='NHWC', reuse=False):
    # bias
    if bias:
        outputs = tf.contrib.layers.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                                           data_format=d_format, rate=rate, activation_fn=None, scope=scope, reuse=reuse)
    else:
        outputs = tf.contrib.layers.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                                           data_format=d_format, rate=rate, activation_fn=None, biases_initializer=None,
                                           scope=scope, reuse=reuse)

    # BN
    if norm:
        outputs = tf.contrib.layers.batch_norm(outputs, decay=0.9, center=True, scale=True, activation_fn=None,
                                               epsilon=1e-5, is_training=is_train, scope=scope + '/batch_norm',
                                               data_format=d_format, reuse=reuse)

    if activation:
        outputs = tf.nn.relu(outputs, name=scope + '/relu')

    return outputs


def max_pool_2d(inputs, kernel_size, scope, stride=2, padding='SAME', data_format='NHWC'):
    outputs = tf.contrib.layers.max_pool2d(inputs, kernel_size, stride=stride,
                                           scope=scope + '/max_pool', padding=padding, data_format=data_format)

    return outputs


def avg_pool_2d(inputs, kernel_size, scope, stride=2, padding='SAME', data_format='NHWC'):
    outputs = tf.contrib.layers.avg_pool2d(inputs, kernel_size, stride=stride,
                                           scope=scope + '/avg_pool', padding=padding, data_format=data_format)

    return outputs
"""
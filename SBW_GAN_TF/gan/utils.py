from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Activation, Lambda
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.layers.convolutional import Conv2D, UpSampling2D
from tensorflow.python.keras.layers.merge import Add
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.pooling import AveragePooling2D
from tensorflow.python.keras.models import Input, Model


def center(inputs, moving_mean, instance_norm=False):
    _, w, h, c = K.int_shape(inputs)
    if instance_norm:
        x_t = tf.transpose(inputs, (0, 3, 1, 2))
        x_flat = tf.reshape(x_t, (-1, c, w * h))
        # (bs, c, w*h)
        m = tf.reduce_mean(x_flat, axis=2, keepdims=True)
        # (bs, c, 1)
    else:
        x_t = tf.transpose(inputs, (3, 0, 1, 2))
        x_flat = tf.reshape(x_t, (c, -1))
        # (c, bs*w*h)
        m = tf.reduce_mean(x_flat, axis=1, keepdims=True)
        m = K.in_train_phase(m, moving_mean)
        # (c, 1)
    f = x_flat - m
    return m, f


def get_decomposition(decomposition, batch_size, group, instance_norm, iter_num, epsilon, device='cpu'):
    if device == 'cpu':
        device = '/cpu:0'
    else:
        device = '/gpu:0'
    if decomposition == 'cholesky' or decomposition == 'cholesky_wm':
        def get_inv_sqrt(ff, m_per_group):
            with tf.device(device):
                sqrt = tf.linalg.cholesky(ff)
            if instance_norm:
                inv_sqrt = tf.linalg.triangular_solve(sqrt,
                                                      tf.tile(tf.expand_dims(tf.expand_dims(tf.eye(m_per_group), 0), 0),
                                                              [batch_size, group, 1, 1]))
            else:
                inv_sqrt = tf.linalg.triangular_solve(sqrt, tf.tile(tf.expand_dims(tf.eye(m_per_group), 0),
                                                                    [group, 1, 1]))
            return sqrt, inv_sqrt
    elif decomposition == 'zca' or decomposition == 'zca_wm':
        def get_inv_sqrt(ff, m_per_group):
            with tf.device(device):
                S, U, _ = tf.svd(ff + tf.eye(m_per_group) * epsilon, full_matrices=True)
            D = tf.linalg.diag(tf.pow(S, -0.5))
            inv_sqrt = tf.matmul(tf.matmul(U, D), U, transpose_b=True)
            D = tf.linalg.diag(tf.pow(S, 0.5))
            sqrt = tf.matmul(tf.matmul(U, D), U, transpose_b=True)
            return sqrt, inv_sqrt
    elif decomposition == 'pca' or decomposition == 'pca_wm':
        def get_inv_sqrt(ff, m_per_group):
            with tf.device(device):
                S, U, _ = tf.svd(ff + tf.eye(m_per_group) * epsilon, full_matrices=True)
            D = tf.linalg.diag(tf.pow(S, -0.5))
            inv_sqrt = tf.matmul(D, U, transpose_b=True)
            D = tf.linalg.diag(tf.pow(S, 0.5))
            sqrt = tf.matmul(D, U, transpose_b=True)
            return sqrt, inv_sqrt
    elif decomposition == 'iter_norm' or decomposition == 'iter_norm_wm':
        def get_inv_sqrt(ff, m_per_group):
            trace = tf.linalg.trace(ff)
            trace = tf.expand_dims(trace, [-1])
            trace = tf.expand_dims(trace, [-1])
            sigma_norm = ff / trace

            projection = tf.eye(m_per_group)
            projection = tf.expand_dims(projection, 0)
            projection = tf.tile(projection, [group, 1, 1])
            for i in range(iter_num):
                projection = (3 * projection - projection * projection * projection * sigma_norm) / 2

            return None, projection / tf.sqrt(trace)
    else:
        assert False
    return get_inv_sqrt


def get_group_cov(inputs, group, m_per_group, instance_norm, bs, w, h, c):
    ff_aprs = []
    for i in range(group):
        start_index = i * m_per_group
        end_index = np.min(((i + 1) * m_per_group, c))
        if instance_norm:
            centered = inputs[:, start_index:end_index, :]
        else:
            centered = inputs[start_index:end_index, :]
        ff_apr = tf.matmul(centered, centered, transpose_b=True)
        ff_apr = tf.expand_dims(ff_apr, 0)
        ff_aprs.append(ff_apr)

    ff_aprs = tf.concat(ff_aprs, 0)
    if instance_norm:
        ff_aprs /= (tf.cast(w * h, tf.float32) - 1.)
    else:
        ff_aprs /= (tf.cast(bs * w * h, tf.float32) - 1.)
    return ff_aprs


def get_group_cov2(inputs, group, m_per_group, instance_norm, bs, w, h, c):
    if instance_norm:
        splits = tf.split(inputs, group, axis=1)  # (bs,m,w*h)
    else:
        splits = tf.split(inputs, group, axis=0)  # (m,bs*w*h)
    centereds = []
    for split in splits:
        centereds.append(tf.expand_dims(split, 0))
    centereds = tf.concat(centereds, 0)  # (group,bs,m,w*h) / (group,m,bs*w*h)
    ff_aprs = tf.matmul(centereds, centereds, transpose_b=True)

    if instance_norm:
        ff_aprs /= (tf.cast(w * h, tf.float32) - 1.)
    else:
        ff_aprs /= (tf.cast(bs * w * h, tf.float32) - 1.)
    return ff_aprs


def get_m_group(x, m, axis):
    channel = K.int_shape(x)[axis]
    if m > 0:
        group = channel // m
    else:
        group = 1
        m = channel
    return group, m


def jacobian(y_flat, x):
    n = y_flat.shape[0]

    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(tf.float32, size=n),
    ]

    _, jacobian = tf.while_loop(
        lambda j, _: j < n,
        lambda j, result: (j+1, result.write(j, tf.gradients(y_flat[j], x))),
        loop_vars)

    return jacobian.stack()


def content_features_model(image_size, layer_name='block4_conv1'):
    from tensorflow.python.keras.applications import vgg19
    x = Input(list(image_size) + [3])
    def preprocess_for_vgg(x):
        x = 255 * (x + 1) / 2
        mean = np.array([103.939, 116.779, 123.68])
        mean = mean.reshape((1, 1, 1, 3))
        x = x - mean
        x = x[..., ::-1]
        return x

    x = Input((128, 64, 3))
    y = Lambda(preprocess_for_vgg)(x)
    vgg = vgg19.VGG19(weights='imagenet', include_top=False, input_tensor=y)
    outputs_dict = dict([(layer.name, layer.output) for layer in vgg.layers])
    if type(layer_name) == list:
        y = [outputs_dict[ln] for ln in layer_name]
    else:
        y = outputs_dict[layer_name]
    return Model(inputs=x, outputs=y)


def uniform_init(shape, constant=4.0, dtype='float32', partition_info=None):
    if len(shape) >= 4:
        stdev = np.sqrt(constant / ((shape[1] ** 2) * (shape[-1] + shape[-2])))
    else:
        stdev = np.sqrt(constant / (shape[0] + shape[1]))
    return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=shape
            ).astype('float32')


he_init = partial(uniform_init, constant=4.0)
glorot_init = partial(uniform_init, constant=2.0)


def resblock(x, kernel_size, resample, nfilters, name, norm=BatchNormalization, is_first=False, conv_layer=Conv2D):
    assert resample in ["UP", "SAME", "DOWN"]

    feature_axis = 1 if K.image_data_format() == 'channels_first' else -1

    identity = lambda x: x

    if norm is None:
        norm = lambda axis, name: identity

    if resample == "UP":
        resample_op = UpSampling2D(size=(2, 2), name=name + '_up')
    elif resample == "DOWN":
        resample_op = AveragePooling2D(pool_size=(2, 2), name=name + '_pool')
    else:
        resample_op = identity

    in_filters = K.int_shape(x)[feature_axis]

    if resample == "SAME" and in_filters == nfilters:
        shortcut_layer = identity
    else:
        shortcut_layer = conv_layer(kernel_size=(1, 1), filters=int(nfilters), kernel_initializer=he_init, name=name + 'shortcut')

    ### SHORTCUT PAHT
    if is_first:
        shortcut = resample_op(x)
        shortcut = shortcut_layer(shortcut)
    else:
        shortcut = shortcut_layer(x)
        shortcut = resample_op(shortcut)

    ### CONV PATH
    convpath = x
    if not is_first:
        convpath = norm(axis=feature_axis, name=name + '_bn1')(convpath)
        convpath = Activation('relu')(convpath)
    if resample == "UP":
        convpath = resample_op(convpath)

    convpath = conv_layer(filters=int(nfilters), kernel_size=kernel_size, kernel_initializer=he_init,
                                      use_bias=True, padding='same', name=name + '_conv1')(convpath)

    convpath = norm(axis=feature_axis, name=name + '_bn2')(convpath)
    convpath = Activation('relu')(convpath)

    convpath = conv_layer(filters=int(nfilters), kernel_size=kernel_size, kernel_initializer=he_init,
                          use_bias=True, padding='same', name=name + '_conv2')(convpath)

    if resample == "DOWN":
        convpath = resample_op(convpath)

    y = Add()([shortcut, convpath])

    return y


def dcblock(x, kernel_size, resample, nfilters, name, norm=BatchNormalization, is_first=False, conv_layer=Conv2D, before_conv=0):
    assert resample in ["UP", "SAME", "DOWN"]

    feature_axis = 1 if K.image_data_format() == 'channels_first' else -1
    nfilters = int(nfilters)
    convpath = x
    if resample == "UP":
        convpath = norm(axis=feature_axis, name=name + '.bn')(convpath)
        convpath = Activation('relu', name=name + 'relu')(convpath)
        convpath = conv_layer(filters=nfilters, kernel_size=kernel_size, strides=(2, 2),
                              name=name + '.conv', padding='same')(convpath)
    elif resample == "SAME":
       if not is_first:
           convpath = norm(axis=feature_axis, name=name + '.bn')(convpath)
           if before_conv == 0:
                convpath = LeakyReLU(name=name + 'relu')(convpath)

       convpath = conv_layer(filters=nfilters, kernel_size=kernel_size,
                             name=name + '.conv', padding='same')(convpath)
       if before_conv != 0:
           convpath = LeakyReLU(name=name + 'relu')(convpath)
    elif resample == "DOWN":
        if not is_first:
            convpath = norm(axis=feature_axis, name=name + '.bn')(convpath)
            if before_conv == 0:
                convpath = LeakyReLU(name=name + 'relu')(convpath)

        convpath = conv_layer(filters=nfilters, kernel_size=kernel_size, strides=(2, 2),
                              name=name + '.conv', padding='same')(convpath)
        if before_conv != 0:
            convpath = LeakyReLU(name=name + 'relu')(convpath)
    return convpath

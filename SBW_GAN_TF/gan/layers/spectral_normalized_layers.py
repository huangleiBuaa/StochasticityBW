import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import Conv2D, Dense, Embedding

from layers.coloring import ConditionalConv11, ConditionalDense, ConditionalDepthwiseConv2D, ConditionalConv2D, FactorizedConv11


def max_singular_val(w, u, fully_differentiable=False, ip=1):
    if not fully_differentiable:
        w_ = K.stop_gradient(w)
    else:
        w_ = w
    u = K.expand_dims(u, axis=-1)

    u_bar = u
    for _ in range(ip):
        v_bar = tf.matmul(w_, u_bar , transpose_a=True)
        v_bar = K.l2_normalize(v_bar, axis=(-1, -2))

        u_bar_raw = tf.matmul(w_, v_bar)
        u_bar = K.l2_normalize(u_bar_raw, axis=(-1, -2))
    sigma = tf.matmul(u_bar, tf.matmul(w, v_bar), transpose_a=True)

    sigma = K.squeeze(sigma, axis=-1)
    sigma = K.squeeze(sigma, axis=-1)

    u_bar = K.squeeze(u_bar, axis=-1)
    return sigma, u_bar


def max_singular_val_for_convolution(w, u, fully_differentiable=False, ip=1, padding='same',
                                     strides=(1, 1), data_format='channels_last'):
    assert ip >= 1
    if not fully_differentiable:
        w_ = K.stop_gradient(w)
    else:
        w_ = w

    u_bar = u
    for _ in range(ip):
        v_bar = K.conv2d(u_bar, w_, strides=strides, data_format=data_format, padding=padding)
        v_bar = K.l2_normalize(v_bar)

        u_bar_raw = K.conv2d_transpose(v_bar, w_, output_shape=K.int_shape(u),
                                       strides=strides, data_format=data_format, padding=padding)
        u_bar = K.l2_normalize(u_bar_raw)

    u_bar_raw_diff = K.conv2d_transpose(v_bar, w, output_shape=K.int_shape(u),
                                        strides=strides, data_format=data_format, padding=padding)
    sigma = K.sum(u_bar * u_bar_raw_diff)
    return sigma, u_bar


class SNConv2D(Conv2D):
    def __init__(self, sigma_initializer=RandomNormal(0, 1), conv_singular=False,
                 fully_diff_spectral=True, spectral_iterations=1, stateful=False, **kwargs):
        super(SNConv2D, self).__init__(**kwargs)
        self.sigma_initializer = keras.initializers.get(sigma_initializer)
        self.conv_singular = conv_singular
        self.fully_diff_spectral = fully_diff_spectral
        self.spectral_iterations = spectral_iterations
        self.stateful = stateful

    def build(self, input_shape):
        super(SNConv2D, self).build(input_shape)
        kernel_shape = K.int_shape(self.kernel)
        if not self.conv_singular:
            self.u = self.add_weight(
                shape=(kernel_shape[0] * kernel_shape[1] * kernel_shape[2], ),
                name='largest_singular_value',
                initializer=self.sigma_initializer,
                trainable=False)
        else:
            self.u = self.add_weight(
                shape=(1, input_shape[1], input_shape[2], input_shape[3]),
                name='largest_singular_value',
                initializer=self.sigma_initializer,
                trainable=False)

    def call(self, inputs):
        if self.conv_singular:
            sigma, u_bar = max_singular_val_for_convolution(self.kernel, self.u,
                                                            fully_differentiable=self.fully_diff_spectral,
                                                            ip=self.spectral_iterations,
                                                            padding=self.padding,
                                                            strides=self.strides, data_format=self.data_format)
            kernel_sn = self.kernel / sigma
            self.add_update(K.update(self.u, u_bar))
        else:
            kernel_shape = K.int_shape(self.kernel)
            w = K.reshape(self.kernel, (kernel_shape[0] * kernel_shape[1] * kernel_shape[2], kernel_shape[3]))

            sigma, u_bar = max_singular_val(w, self.u, fully_differentiable=self.fully_diff_spectral,
                                            ip=self.spectral_iterations)

            w_sn = w / sigma

            kernel_sn = K.reshape(w_sn, kernel_shape)

            self.add_update(K.update(self.u, u_bar))

        kernel = self.kernel
        self.kernel = kernel_sn
        outputs = super(SNConv2D, self).call(inputs)
        self.kernel = kernel

        return outputs


class SNDense(Dense):
    def __init__(self, sigma_initializer=RandomNormal(0, 1), spectral_iterations=1,
                 fully_diff_spectral=True, stateful=False, **kwargs):
        super(SNDense, self).__init__(**kwargs)
        self.sigma_initializer = keras.initializers.get(sigma_initializer)
        self.fully_diff_spectral = fully_diff_spectral
        self.spectral_iterations = spectral_iterations
        self.stateful = stateful

    def build(self, input_shape):
        super(SNDense, self).build(input_shape)
        kernel_shape = K.int_shape(self.kernel)
        self.u = self.add_weight(
            shape=(kernel_shape[0], ),
            name='largest_singular_value',
            initializer=self.sigma_initializer,
            trainable=False)

    def call(self, inputs):
        w = self.kernel
        sigma, u_bar = max_singular_val(w, self.u, fully_differentiable=self.fully_diff_spectral,
                                        ip=self.spectral_iterations)
        w_sn = w / sigma
        kernel_sn = w_sn
        self.add_update(K.update(self.u, u_bar))

        kernel = self.kernel
        self.kernel = kernel_sn
        outputs = super(SNDense, self).call(inputs)
        self.kernel = kernel

        return outputs


class SNEmbeding(Embedding):
    def __init__(self, sigma_initializer=RandomNormal(0, 1), spectral_iterations=1,
                 fully_diff_spectral=True, stateful=False, **kwargs):
        super(SNEmbeding, self).__init__(**kwargs)
        self.sigma_initializer = keras.initializers.get(sigma_initializer)
        self.fully_diff_spectral = fully_diff_spectral
        self.spectral_iterations = spectral_iterations
        self.stateful = stateful

    def build(self, input_shape):
        super(SNEmbeding, self).build(input_shape)
        kernel_shape = K.int_shape(self.embeddings)
        self.u = self.add_weight(
            shape=(kernel_shape[0], ),
            name='largest_singular_value',
            initializer=self.sigma_initializer,
            trainable=False)

    def call(self, inputs):
        w = self.embeddings
        sigma, u_bar = max_singular_val(w, self.u, fully_differentiable=self.fully_diff_spectral,
                                        ip=self.spectral_iterations)
        w_sn = w / sigma
        kernel_sn = w_sn
        self.add_update(K.update(self.u, u_bar))

        embeddings = self.embeddings
        self.embeddings = kernel_sn
        outputs = super(SNEmbeding, self).call(inputs)
        self.embeddings = embeddings

        return outputs


class SNConditionalConv11(ConditionalConv11):
    def __init__(self, sigma_initializer=RandomNormal(0, 1), spectral_iterations=1,
                 fully_diff_spectral=True,  stateful=False, renormalize=False, **kwargs):
        """
        renormalize - if True compute only one sigma for kernel, otherwise compute sigma per class
        """
        super(SNConditionalConv11, self).__init__(**kwargs)
        self.sigma_initializer = keras.initializers.get(sigma_initializer)
        self.fully_diff_spectral = fully_diff_spectral
        self.spectral_iterations = spectral_iterations
        self.stateful = stateful
        self.renormalize = renormalize

    def build(self, input_shape):
        super(SNConditionalConv11, self).build(input_shape)
        kernel_shape = K.int_shape(self.kernel)
        if not self.renormalize:
            self.u = self.add_weight(
                shape=(self.number_of_classes, kernel_shape[1] * kernel_shape[2] * kernel_shape[3]),
                name='largest_singular_value',
                initializer=self.sigma_initializer,
                trainable=False)
        else:
            self.u = self.add_weight(
                shape=(self.number_of_classes * kernel_shape[1] * kernel_shape[2] * kernel_shape[3], ),
                name='largest_singular_value',
                initializer=self.sigma_initializer,
                trainable=False)

    def call(self, inputs):
        kernel_shape = K.int_shape(self.kernel)
        if not self.renormalize:
            w = K.reshape(self.kernel, (kernel_shape[0], kernel_shape[1] * kernel_shape[2] * kernel_shape[3], kernel_shape[-1]))
            sigma, u_bar = max_singular_val(w, self.u,
                                            fully_differentiable=self.fully_diff_spectral, ip=self.spectral_iterations)
            sigma = K.reshape(sigma, (self.number_of_classes, 1, 1, 1, 1))
        else:
            w = K.reshape(self.kernel, (-1, kernel_shape[-1]))
            sigma, u_bar = max_singular_val(w, self.u,
                                            fully_differentiable=self.fully_diff_spectral, ip=self.spectral_iterations)

 
        self.add_update(K.update(self.u, u_bar))

        kernel = self.kernel
        self.kernel = self.kernel / sigma
        outputs = super(SNConditionalConv11, self).call(inputs)
        self.kernel = kernel

        return outputs


class SNFactorizedConv11(FactorizedConv11):
    def __init__(self, sigma_initializer=RandomNormal(0, 1), spectral_iterations=1,
                 fully_diff_spectral=True, stateful=False, renormalize=False, **kwargs):
        """
        renormalize - if True compute only one sigma for kernel, otherwise compute sigma per class
        """
        super(SNFactorizedConv11, self).__init__(**kwargs)
        self.sigma_initializer = keras.initializers.get(sigma_initializer)
        self.fully_diff_spectral = fully_diff_spectral
        self.spectral_iterations = spectral_iterations
        self.stateful = stateful
        self.renormalize = renormalize

    def build(self, input_shape):
        super(SNFactorizedConv11, self).build(input_shape)
        kernel_shape = K.int_shape(self.kernel)
        if not self.renormalize:
            self.u = self.add_weight(
                shape=(self.filters_emb, kernel_shape[1] * kernel_shape[2] * kernel_shape[3]),
                name='largest_singular_value',
                initializer=self.sigma_initializer,
                trainable=False)
        else:
            self.u = self.add_weight(
                shape=(self.filters_emb * kernel_shape[1] * kernel_shape[2] * kernel_shape[3], ),
                name='largest_singular_value',
                initializer=self.sigma_initializer,
                trainable=False)

    def call(self, inputs):
        kernel_shape = K.int_shape(self.kernel)
        if not self.renormalize:
            w = K.reshape(self.kernel, (kernel_shape[0], kernel_shape[1] * kernel_shape[2] * kernel_shape[3], kernel_shape[-1]))
            sigma, u_bar = max_singular_val(w, self.u,
                                            fully_differentiable=self.fully_diff_spectral, ip=self.spectral_iterations)
            sigma = K.reshape(sigma, (self.filters_emb, 1, 1, 1, 1))
        else:
            w = K.reshape(self.kernel, (-1, kernel_shape[-1]))
            sigma, u_bar = max_singular_val(w, self.u,
                                            fully_differentiable=self.fully_diff_spectral, ip=self.spectral_iterations)

        self.add_update(K.update(self.u, u_bar))

        kernel = self.kernel
        self.kernel = self.kernel / sigma
        outputs = super(SNFactorizedConv11, self).call(inputs)
        self.kernel = kernel

        return outputs


class SNConditionalConv2D(ConditionalConv2D):
    def __init__(self, sigma_initializer=RandomNormal(0, 1), spectral_iterations=1,
                 fully_diff_spectral=True,  stateful=False, renormalize=False, **kwargs):
        """
        renormalize - if True compute only one sigma for kernel, otherwise compute sigma per class
        """
        super(SNConditionalConv2D, self).__init__(**kwargs)
        self.sigma_initializer = keras.initializers.get(sigma_initializer)
        self.fully_diff_spectral = fully_diff_spectral
        self.spectral_iterations = spectral_iterations
        self.stateful = stateful
        self.renormalize = renormalize

    def build(self, input_shape):
        super(SNConditionalConv2D, self).build(input_shape)
        kernel_shape = K.int_shape(self.kernel)
        if not self.renormalize:
            self.u = self.add_weight(
                shape=(self.number_of_classes, kernel_shape[1] * kernel_shape[2] * kernel_shape[3]),
                name='largest_singular_value',
                initializer=self.sigma_initializer,
                trainable=False)
        else:
            self.u = self.add_weight(
                shape=(self.number_of_classes * kernel_shape[1] * kernel_shape[2] * kernel_shape[3], ),
                name='largest_singular_value',
                initializer=self.sigma_initializer,
                trainable=False)

    def call(self, inputs):
        kernel_shape = K.int_shape(self.kernel)
        if not self.renormalize:
            w = K.reshape(self.kernel, (kernel_shape[0], kernel_shape[1] * kernel_shape[2] * kernel_shape[3], kernel_shape[-1]))
            sigma, u_bar = max_singular_val(w, self.u,
                                            fully_differentiable=self.fully_diff_spectral, ip=self.spectral_iterations)
            sigma = K.reshape(sigma, (self.number_of_classes, 1, 1, 1, 1))
        else:
            w = K.reshape(self.kernel, (-1, kernel_shape[-1]))
            sigma, u_bar = max_singular_val(w, self.u,
                                            fully_differentiable=self.fully_diff_spectral, ip=self.spectral_iterations)

        self.add_update(K.update(self.u, u_bar))

        kernel = self.kernel
        self.kernel = self.kernel / sigma
        outputs = super(SNConditionalConv2D, self).call(inputs)
        self.kernel = kernel

        return outputs


class SNConditionalDepthwiseConv2D(ConditionalDepthwiseConv2D):
    def __init__(self, sigma_initializer=RandomNormal(0, 1), spectral_iterations=1,
                 fully_diff_spectral=True,  stateful=False, renormalize=True, **kwargs):
        super(SNConditionalDepthwiseConv2D, self).__init__(**kwargs)
        self.sigma_initializer = keras.initializers.get(sigma_initializer)
        self.fully_diff_spectral = fully_diff_spectral
        self.spectral_iterations = spectral_iterations
        self.stateful = stateful
        self.renormalize = renormalize

    def build(self, input_shape):
        super(SNConditionalDepthwiseConv2D, self).build(input_shape)
        kernel_shape = K.int_shape(self.kernel)
        if self.renormalize:
            self.u = self.add_weight(
                shape=(kernel_shape[0] * kernel_shape[1] * kernel_shape[2], ),
                name='largest_singular_value',
                initializer=self.sigma_initializer,
                trainable=False)
        else:
            self.u = self.add_weight(
                shape=(kernel_shape[0] * kernel_shape[3], kernel_shape[1] * kernel_shape[2]),
                name='largest_singular_value',
                initializer=self.sigma_initializer,
                trainable=False)

    def call(self, inputs):
        kernel_shape = K.int_shape(self.kernel)

        if self.renormalize:
            w = K.reshape(self.kernel, (-1, kernel_shape[-1]))

            sigma, u_bar = max_singular_val(w, self.u,
                                            fully_differentiable=self.fully_diff_spectral, ip=self.spectral_iterations)
        else:
            w = tf.transpose(self.kernel, (0, 3, 1, 2))
            w = K.reshape(w, [-1, kernel_shape[1] * kernel_shape[2]])
            w = K.expand_dims(w, axis=-1)
            sigma, u_bar = max_singular_val(w, self.u,
                                            fully_differentiable=self.fully_diff_spectral, ip=self.spectral_iterations)

            sigma = K.reshape(sigma, [kernel_shape[0], 1, 1, kernel_shape[-1]])

        self.add_update(K.update(self.u, u_bar))

        kernel = self.kernel
        self.kernel = self.kernel / sigma
        outputs = super(SNConditionalDepthwiseConv2D, self).call(inputs)
        self.kernel = kernel

        return outputs


class SNCondtionalDense(ConditionalDense):
    def __init__(self, sigma_initializer=RandomNormal(0, 1), spectral_iterations=1,
                 fully_diff_spectral=True,  stateful=False, renormalize=False, **kwargs):
        """
        renormalize - if True compute only one sigma for kernel, otherwise compute sigma per class
        """
        super(SNCondtionalDense, self).__init__(**kwargs)
        self.sigma_initializer = keras.initializers.get(sigma_initializer)
        self.fully_diff_spectral = fully_diff_spectral
        self.spectral_iterations = spectral_iterations
        self.stateful = stateful
        self.renormalize = renormalize

    def build(self, input_shape):
        super(SNCondtionalDense, self).build(input_shape)
        kernel_shape = K.int_shape(self.kernel)
        if not self.renormalize:
            self.u = self.add_weight(
                shape=(self.number_of_classes, kernel_shape[1]),
                name='largest_singular_value',
                initializer=self.sigma_initializer,
                trainable=False)
        else:
            self.u = self.add_weight(
                shape=(self.number_of_classes * kernel_shape[1], ),
                name='largest_singular_value',
                initializer=self.sigma_initializer,
                trainable=False)

    def call(self, inputs):
        w = self.kernel
        kernel_shape = K.int_shape(self.kernel)
        if self.renormalize:
            w = K.reshape(w, [-1, kernel_shape[-1]])
            sigma, u_bar = max_singular_val(w, self.u,
                                            fully_differentiable=self.fully_diff_spectral, ip=self.spectral_iterations)
        else:
            sigma, u_bar = max_singular_val(w, self.u,
                                            fully_differentiable=self.fully_diff_spectral, ip=self.spectral_iterations)
            sigma = K.reshape(sigma, (self.number_of_classes, 1, 1))

        self.add_update(K.update(self.u, u_bar))

        kernel = self.kernel
        self.kernel = self.kernel / sigma
        outputs = super(SNCondtionalDense, self).call(inputs)
        self.kernel = kernel

        return outputs

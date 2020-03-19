import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers, regularizers, constraints
from tensorflow.python.keras.layers import Layer
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.keras.utils import conv_utils

sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./gan'))
import utils


class ConditionalInstanceNormalization(Layer):
    """Conditional Instance normalization layer.
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    Each class has it own normalization parametes.
    # Arguments
        number_of_classes: Number of classes, 10 for cifar10.
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [A Learned Representation For Artistic Style](https://arxiv.org/abs/1610.07629)
    """
    def __init__(self,
                 number_of_classes,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(ConditionalInstanceNormalization, self).__init__(**kwargs)
        self.number_of_classes = number_of_classes
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape[0])
        cls = input_shape[1]
        if len(cls) != 2:
            raise ValueError("Classes should be one dimensional")

        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        if self.axis is None:
            shape = (self.number_of_classes, 1)
        else:
            shape = (self.number_of_classes, input_shape[0][self.axis])

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        super(ConditionalInstanceNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        class_labels = K.squeeze(inputs[1], axis=1)
        inputs = inputs[0]
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[0] = K.shape(inputs)[0]
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(K.gather(self.gamma, class_labels), broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(K.gather(self.beta, class_labels), broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = {
            'number_of_classes': self.number_of_classes,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(ConditionalInstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConditionalBatchNormalization(Layer):
    """Batch normalization layer (Ioffe and Szegedy, 2014).
    Normalize the activations of the previous layer at each batch,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `BatchNormalization`.
        momentum: Momentum for the moving average.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        moving_mean_initializer: Initializer for the moving mean.
        moving_variance_initializer: Initializer for the moving variance.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
    """
    def __init__(self,
                 number_of_classes,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(ConditionalBatchNormalization, self).__init__(**kwargs)
        self.number_of_classes = number_of_classes
        self.supports_masking = True
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(moving_variance_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        input_shape = input_shape[0]
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        shape = (dim, )

        if self.scale:
            self.gamma = self.add_weight((self.number_of_classes, dim),
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight((self.number_of_classes, dim),
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.moving_mean = self.add_weight(
            shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)
        self.moving_variance = self.add_weight(
            shape,
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        class_labels = K.squeeze(inputs[1], axis=1)
        inputs = inputs[0]
        input_shape = K.int_shape(inputs)
        # Prepare broadcasting shape.
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        # Determines whether broadcasting is needed.
        needs_broadcasting = (sorted(reduction_axes) != range(ndim)[:-1])

        param_broadcast = [1] * len(input_shape)
        param_broadcast[self.axis] = input_shape[self.axis]
        param_broadcast[0] = K.shape(inputs)[0]
        if self.scale:
            broadcast_gamma = K.reshape(K.gather(self.gamma, class_labels), param_broadcast)
        else:
            broadcast_gamma = None

        if self.center:
            broadcast_beta = K.reshape(K.gather(self.beta, class_labels), param_broadcast)
        else:
            broadcast_beta = None

        normed, mean, variance = K.normalize_batch_in_training(
            inputs, gamma=None, beta=None,
            reduction_axes=reduction_axes, epsilon=self.epsilon)

        if training in {0, False}:
            return normed
        else:
            self.add_update([K.moving_average_update(self.moving_mean,
                                                     mean,
                                                     self.momentum),
                             K.moving_average_update(self.moving_variance,
                                                     variance,
                                                     self.momentum)],
                            inputs)

            def normalize_inference():
                if needs_broadcasting:
                    # In this case we must explictly broadcast all parameters.
                    broadcast_moving_mean = K.reshape(self.moving_mean,
                                                      broadcast_shape)
                    broadcast_moving_variance = K.reshape(self.moving_variance,
                                                          broadcast_shape)
                    return K.batch_normalization(
                        inputs,
                        broadcast_moving_mean,
                        broadcast_moving_variance,
                        beta=None,
                        gamma=None,
                        epsilon=self.epsilon)
                else:
                    return K.batch_normalization(
                        inputs,
                        self.moving_mean,
                        self.moving_variance,
                        beta=None,
                        gamma=None,
                        epsilon=self.epsilon)

        # Pick the normalized form corresponding to the training phase.
        out = K.in_train_phase(normed,
                                normalize_inference,
                                training=training)
        return out * broadcast_gamma + broadcast_beta

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = {
            'number_of_classes': self.number_of_classes,
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer': initializers.serialize(self.moving_variance_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(ConditionalBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DecorelationNormalization(Layer):
    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 m_per_group=0,
                 decomposition='cholesky',
                 iter_num=5,
                 instance_norm=0,
                 renorm=False,
                 data_format=None,
                 moving_mean_initializer='zeros',
                 moving_cov_initializer='identity',
                 device='cpu',
                 **kwargs):
        assert decomposition in ['cholesky', 'zca', 'pca', 'iter_norm',
                                 'cholesky_wm', 'zca_wm', 'pca_wm', 'iter_norm_wm']
        super(DecorelationNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.momentum = momentum
        self.epsilon = epsilon
        self.m_per_group = m_per_group
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        # self.moving_cov_initializer = initializers.get(moving_cov_initializer)
        self.axis = axis
        self.renorm = renorm
        self.decomposition = decomposition
        self.iter_num = iter_num
        self.instance_norm = instance_norm
        self.device = device
        self.data_format = conv_utils.normalize_data_format(data_format)

    def matrix_initializer(self, shape, dtype=tf.float32, partition_info=None):
        moving_convs = []
        for i in range(shape[0]):
            moving_conv = tf.expand_dims(tf.eye(shape[1], dtype=dtype), 0)
            moving_convs.append(moving_conv)

        moving_convs = tf.concat(moving_convs, 0)
        return moving_convs

    def build(self, input_shape):
        assert self.data_format == 'channels_last'
        dim = input_shape.as_list()[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if self.m_per_group == 0:
            self.m_per_group = dim
        self.group = dim // self.m_per_group
        assert (dim % self.m_per_group == 0), 'dim is {}, m is {}'.format(dim, self.m_per_group)

        self.moving_mean = self.add_weight(shape=(dim, 1),
                                           name='moving_mean',
                                           synchronization=tf_variables.VariableSynchronization.ON_READ,
                                           initializer=self.moving_mean_initializer,
                                           trainable=False,
                                           aggregation=tf_variables.VariableAggregation.MEAN)
        self.moving_matrix = self.add_weight(shape=(self.group, self.m_per_group, self.m_per_group),
                                             name='moving_matrix',
                                             synchronization=tf_variables.VariableSynchronization.ON_READ,
                                             initializer=self.matrix_initializer,
                                             trainable=False,
                                             aggregation=tf_variables.VariableAggregation.MEAN)

        self.built = True

    def call(self, inputs, training=None):
        _, w, h, c = K.int_shape(inputs)
        bs = K.shape(inputs)[0]

        m, f = utils.center(inputs, self.moving_mean, self.instance_norm)
        get_inv_sqrt = utils.get_decomposition(self.decomposition, bs, self.group, self.instance_norm, self.iter_num, self.epsilon, self.device)

        def train():
            ff_aprs = utils.get_group_cov(f, self.group, self.m_per_group, self.instance_norm, bs, w, h, c)

            if self.instance_norm:
                ff_aprs = tf.transpose(ff_aprs, (1, 0, 2, 3))
                ff_aprs = (1 - self.epsilon) * ff_aprs + tf.expand_dims(tf.expand_dims(tf.eye(self.m_per_group) * self.epsilon, 0), 0)
            else:
                ff_aprs = (1 - self.epsilon) * ff_aprs + tf.expand_dims(tf.eye(self.m_per_group) * self.epsilon, 0)

            whitten_matrix = get_inv_sqrt(ff_aprs, self.m_per_group)[1]

            self.add_update([K.moving_average_update(self.moving_mean,
                                                     m,
                                                     self.momentum),
                             K.moving_average_update(self.moving_matrix,
                                                     whitten_matrix if '_wm' in self.decomposition else ff_aprs,
                                                     self.momentum)],
                            inputs)

            if self.renorm:
                l, l_inv = get_inv_sqrt(ff_aprs, self.m_per_group)
                ff_mov = (1 - self.epsilon) * self.moving_matrix + tf.eye(self.m_per_group) * self.epsilon
                _, l_mov_inverse = get_inv_sqrt(ff_mov, self.m_per_group)
                l_ndiff = K.stop_gradient(l)
                return tf.matmul(tf.matmul(l_mov_inverse, l_ndiff), l_inv)

            return whitten_matrix

        def test():
            moving_matrix = (1 - self.epsilon) * self.moving_matrix + tf.eye(self.m_per_group) * self.epsilon
            if '_wm' in self.decomposition:
                return moving_matrix
            else:
                return get_inv_sqrt(moving_matrix, self.m_per_group)[1]

        if self.instance_norm == 1:
            inv_sqrt = train()
            f = tf.reshape(f, [-1, self.group, self.m_per_group, w*h])
            f_hat = tf.matmul(inv_sqrt, f)
            decorelated = K.reshape(f_hat, [bs, c, w, h])
            decorelated = tf.transpose(decorelated, [0, 2, 3, 1])
        else:
            inv_sqrt = K.in_train_phase(train, test)
            f = tf.reshape(f, [self.group, self.m_per_group, -1])
            f_hat = tf.matmul(inv_sqrt, f)
            decorelated = K.reshape(f_hat, [c, bs, w, h])
            decorelated = tf.transpose(decorelated, [1, 2, 3, 0])

        return decorelated

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
            'moving_matrix_initializer': initializers.serialize(self.matrix_initializer)
        }
        base_config = super(DecorelationNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def test_dbn_eager():
    tf.enable_eager_execution()
    data = tf.random.normal([1, 7, 7, 16])
    K.set_learning_phase(1)
    # tf.set_random_seed(1)
    decor1 = DecorelationNormalization(m_per_group=2, decomposition='zca', instance_norm=1)
    decor2 = DecorelationNormalization(m_per_group=2, decomposition='zca', instance_norm=0)
    decor3 = DecorelationNormalization(m_per_group=2, decomposition='zca', instance_norm=0, group_conv=2)
    out1 = decor1(data)
    out2 = decor2(data)
    out3 = decor3(data)
    out1 = tf.reduce_sum(out1)
    out2 = tf.reduce_sum(out2)
    out3 = tf.reduce_sum(out3)
    print(out1)
    print(out2)
    print(out3)


def test_dbn_eager2():
    tf.enable_eager_execution()
    data = tf.random.normal([1, 7, 7, 16])
    K.set_learning_phase(1)
    # tf.set_random_seed(1)
    decor1 = DecorelationNormalization(m_per_group=2, decomposition='pca', instance_norm=0)
    decor2 = DecorelationNormalization(m_per_group=2, decomposition='pca', instance_norm=1)
    import time
    out1 = decor1(data)
    t1 = time.time()
    out1 = decor1(data)
    t2 = time.time()
    out2 = decor2(data)
    t3 = time.time()
    distance = np.sum(np.square(out1-out2))
    print(distance)
    print('t1:', t2 - t1)
    print('t2:', t3 - t2)


def test_dbn2():
    # tf.enable_eager_execution()
    inputs = tf.keras.Input((7, 7, 16))
    data = np.random.normal(0, 1, [1, 7, 7, 16])
    data2 = np.random.normal(0, 1, [256, 7, 7, 16])
    K.set_learning_phase(1)
    # tf.set_random_seed(1)
    decor1 = DecorelationNormalization(m_per_group=2, decomposition='zca', instance_norm=0)
    decor2 = DecorelationNormalization(m_per_group=2, decomposition='zca', instance_norm=1)
    out1, out2 = decor1(inputs), decor2(inputs)
    op1 = K.function(inputs, out1)
    op2 = K.function(inputs, out2)
    import time
    out2 = op2(data2)
    t1 = time.time()
    out1 = op1(data)
    t2 = time.time()
    out2 = op2(data)
    t3 = time.time()
    distance = np.sum(np.square(out1-out2))
    print(distance)
    print('t1:', t2 - t1)
    print('t2:', t3 - t2)


def test_dbn():
    inputs = tf.keras.layers.Input([8, 8, 16])
    data = np.random.normal(0, 1, [128, 8, 8, 16])
    # K.set_learning_phase(1)
    decor = DecorelationNormalization(group=1, instance_norm=1)
    out = decor(inputs)
    out = tf.reduce_mean(out)
    sess = K.get_session()
    sess.run(tf.global_variables_initializer())
    K.set_learning_phase(1)
    outputs = sess.run([out], feed_dict={inputs: data})
    print(np.mean(outputs))


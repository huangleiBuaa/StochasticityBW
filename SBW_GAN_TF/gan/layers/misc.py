import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.layers.pooling import GlobalPooling2D
from tensorflow.python.keras.optimizers import Adam


class ConditionalAdamOptimizer(Adam):
    def __init__(self, lr_decay_schedule=None, **kwargs):
        super(ConditionalAdamOptimizer, self).__init__(**kwargs)
        self.lr_decay_schedule = lr_decay_schedule

        if lr_decay_schedule.startswith('dropatc'):
            drop_at = int(lr_decay_schedule.replace('dropatc', ''))
            drop_at_generator = drop_at * 1000
            self.lr_decay_schedule_generator = lambda iter: tf.where(K.less(iter, drop_at_generator), 1.,  0.1)
        else:
            self.lr_decay_schedule_generator = lambda iter: 1.

    def get_updates(self, loss, params):
        conditional_params = [param for param in params if '_repart_c' in param.name]
        unconditional_params = [param for param in params if '_repart_c' not in param.name]

        # print (conditional_params)
        # print (unconditional_params)

        print(len(params))
        print(len(conditional_params))
        print(len(unconditional_params))

        lr = self.lr
        self.lr = self.lr_decay_schedule_generator(self.iterations) * lr
        updates = super(ConditionalAdamOptimizer, self).get_updates(loss, conditional_params)[1:]
        self.lr = lr
        updates += super(ConditionalAdamOptimizer, self).get_updates(loss, unconditional_params)
        #updates.append(K.update_sub(self.iterations, 1))
        return updates


class Split(Layer):
    def __init__(self, num_or_size_splits, axis, **kwargs):
        super(Split, self).__init__(**kwargs)
        self.num_or_size_splits = num_or_size_splits
        self.axis = axis

    def call(self, inputs):
        splits = tf.split(inputs, self.num_or_size_splits, self.axis)
        return splits


class GlobalSumPooling2D(GlobalPooling2D):
    """Global sum pooling operation for spatial data.
    # Arguments
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Input shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, rows, cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, rows, cols)`
    # Output shape
        2D tensor with shape:
        `(batch_size, channels)`
    """

    def call(self, inputs):
        if self.data_format == 'channels_last':
            return K.sum(inputs, axis=[1, 2])
        else:
            return K.sum(inputs, axis=[2, 3])


class GaussianFromPointsLayer(Layer):
    def __init__(self, sigma=6, image_size=(128, 64), **kwargs):
        self.sigma = sigma
        self.image_size = image_size
        super(GaussianFromPointsLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.xx, self.yy = tf.meshgrid(tf.range(self.image_size[1]),
                                        tf.range(self.image_size[0]))
        self.xx = tf.expand_dims(tf.cast(self.xx, 'float32'), 2)
        self.yy = tf.expand_dims(tf.cast(self.yy, 'float32'), 2)

    def call(self, x, mask=None):
        def batch_map(cords):
            y = ((cords[..., 0] + 1.0) / 2.0) * self.image_size[0]
            x = ((cords[..., 1] + 1.0) / 2.0) * self.image_size[1]
            y = tf.reshape(y, (1, 1, -1))
            x = tf.reshape(x, (1, 1, -1))
            return tf.exp(-((self.yy - y) ** 2 + (self.xx - x) ** 2) / (2 * self.sigma ** 2))

        x = tf.map_fn(batch_map, x, dtype='float32')
        print (x.shape)
        return x

    def compute_output_shape(self, input_shape):
        print (input_shape)
        return tuple([input_shape[0], self.image_size[0], self.image_size[1], input_shape[1]])

    def get_config(self):
        config = {"sigma": self.sigma, "image_size": self.image_size}
        base_config = super(GaussianFromPointsLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


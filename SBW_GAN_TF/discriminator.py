from tensorflow.python.keras.models import Input, Model
from tensorflow.python.keras.layers import Dense, Activation, Conv2D, GlobalAveragePooling2D, Lambda, Dropout, Flatten
from tensorflow.python.keras.layers import Add, Embedding, LeakyReLU

from gan.utils import resblock, dcblock
from gan.layers.coloring import ConditionalConv11
from layers.spectral_normalized_layers import SNConv2D, SNDense, SNConditionalConv11, SNEmbeding
from gan.layers.misc import GlobalSumPooling2D
from gan.utils import glorot_init
from functools import partial
from tensorflow.python.keras import backend as K

from generator import create_norm


def make_discriminator(input_image_shape, input_cls_shape=(1, ), block_sizes=(128, 128, 128, 128),
                       resamples=('DOWN', "DOWN", "SAME", "SAME"),
                       number_of_classes=10, type='AC_GAN',
                       norm='n', decomposition='cholesky', whitten_m=1, coloring_m=1, iter_num=5, instance_norm=0,
                       coloring='n',
                       spectral=False,
                       fully_diff_spectral=False, spectral_iterations=1, conv_singular=True,
                       before_conv=0,
                       sum_pool=False, dropout=False, arch='res', filters_emb=10):

    assert arch in ['res', 'dcgan']
    assert len(block_sizes) == len(resamples)
    x = Input(input_image_shape, name='DInputImage')
    cls = Input(input_cls_shape, dtype='int32', name='DLabel')

    if spectral:
        conv_layer = partial(SNConv2D, conv_singular=conv_singular,
                             fully_diff_spectral=fully_diff_spectral, spectral_iterations=spectral_iterations)
        cond_conv_layer = partial(SNConditionalConv11,
                                  fully_diff_spectral=fully_diff_spectral, spectral_iterations=spectral_iterations)
        dence_layer = partial(SNDense,
                              fully_diff_spectral=fully_diff_spectral, spectral_iterations=spectral_iterations)
        emb_layer = partial(SNEmbeding, fully_diff_spectral=fully_diff_spectral, spectral_iterations=spectral_iterations)
    else:
        conv_layer = Conv2D
        cond_conv_layer = ConditionalConv11
        dence_layer = Dense
        emb_layer = Embedding

    norm_layer = create_norm(norm=norm, coloring=coloring,
                             decomposition=decomposition, iter_num=iter_num, whitten_m=whitten_m, coloring_m=coloring_m, instance_norm=instance_norm,
                             cls=cls, number_of_classes=number_of_classes,
                             conditional_conv_layer=cond_conv_layer, uncoditional_conv_layer=conv_layer,
                             filters_emb=filters_emb)

    y = x
    i = 0
    for block_size, resample in zip(block_sizes, resamples):
        if arch == 'res':
            y = resblock(y, kernel_size=(3, 3), resample=resample, nfilters=block_size,
                         name='Discriminator.' + str(i), norm=norm_layer, is_first=(i == 0), conv_layer=conv_layer)
            i += 1
        else:
            y = dcblock(y, kernel_size=(3, 3) if resample == "SAME" else (4, 4), resample=resample, nfilters=block_size,
                        name='Discriminator.' + str(i), norm=norm_layer, is_first=(i == 0), conv_layer=conv_layer, before_conv=before_conv)
            i += 1

    if arch == 'res':
        y = Activation('relu')(y)
    else:
        y = LeakyReLU()(y)

    if arch == 'res':
        if sum_pool:
            y = GlobalSumPooling2D()(y)
        else:
            y = GlobalAveragePooling2D()(y)
    else:
        y = Flatten()(y)

    if dropout != 0:
        y = Dropout(dropout)(y)

    if type == 'AC_GAN':
        cls_out = Dense(units=number_of_classes, use_bias=True, kernel_initializer=glorot_init)(y)
        out = dence_layer(units=1, use_bias=True, kernel_initializer=glorot_init)(y)
        return Model(inputs=x, outputs=[out, cls_out])
    elif type == "PROJECTIVE":
        emb = emb_layer(input_dim = number_of_classes, output_dim = block_sizes[-1])(cls)
        phi = Lambda(lambda inp: K.sum(inp[1] * K.expand_dims(inp[0], axis=1), axis=2), output_shape=(1, ))([y, emb])
        psi = dence_layer(units=1, use_bias=True, kernel_initializer=glorot_init)(y)
        out = Add()([phi, psi])
        return Model(inputs=[x, cls], outputs=[out])
    elif type is None:
        out = dence_layer(units=1, use_bias=True, kernel_initializer=glorot_init)(y)
        return Model(inputs=[x], outputs=[out])

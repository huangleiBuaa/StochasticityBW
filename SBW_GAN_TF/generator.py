from tensorflow.python.keras.models import Input, Model
from tensorflow.python.keras.layers import Dense, Reshape, Activation, Conv2D, Conv2DTranspose
from tensorflow.python.keras.layers import BatchNormalization, Add, Embedding, Concatenate

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

from gan.utils import glorot_init, resblock, dcblock, get_m_group
from gan.layers.coloring import ConditionalConv11, ConditionalCenterScale, CenterScale, FactorizedConv11
from gan.layers.normalization import DecorelationNormalization
from gan.layers.misc import Split
from layers.spectral_normalized_layers import SNConv2D, SNConditionalConv11, SNDense, SNEmbeding, SNFactorizedConv11
from functools import partial


def create_norm(norm, coloring,
                decomposition='zca', iter_num=5, whitten_m=0, coloring_m=0, instance_norm=0, device='cpu',
                cls=None, number_of_classes=None, filters_emb=10,
                uncoditional_conv_layer=Conv2D, conditional_conv_layer=ConditionalConv11,
                factor_conv_layer=FactorizedConv11):
    assert norm in ['n', 'b', 'd', 'dr']
    assert coloring in ['ucs', 'ccs', 'uccs', 'uconv', 'fconv', 'ufconv', 'cconv', 'ucconv', 'ccsuconv', 'n']

    if norm == 'n':
        norm_layer = lambda axis, name: (lambda inp: inp)
    elif norm == 'b':
        norm_layer = lambda axis, name: BatchNormalization(axis=axis, center=False, scale=False, name=name)
    elif norm == 'd':
        norm_layer = lambda axis, name: DecorelationNormalization(name=name,
                                                                  m_per_group=whitten_m,
                                                                  decomposition=decomposition,
                                                                  iter_num=iter_num,
                                                                  instance_norm=instance_norm,
                                                                  device=device)
    elif norm == 'dr':
        norm_layer = lambda axis, name: DecorelationNormalization(name=name,
                                                                  m_per_group=whitten_m,
                                                                  decomposition=decomposition,
                                                                  iter_num=iter_num,
                                                                  instance_norm=instance_norm,
                                                                  renorm=True)

    if coloring == 'ccs':
        after_norm_layer = lambda axis, name: lambda x: ConditionalCenterScale(number_of_classes=number_of_classes,
                                                                               axis=axis, name=name)([x, cls])
    elif coloring == 'ucs':
        after_norm_layer = lambda axis, name: lambda x: CenterScale(axis=axis, name=name)(x)
    elif coloring == 'uccs':
        def after_norm_layer(axis, name):
            def f(x):
                c = ConditionalCenterScale(number_of_classes=number_of_classes, axis=axis, name=name + '_c')([x, cls])
                u = CenterScale(axis=axis, name=name + '_u')(x)
                out = Add(name=name + '_a')([c, u])
                return out
            return f
    elif coloring == 'cconv':
        def after_norm_layer(axis, name):
            def f(x):
                coloring_group, m = get_m_group(x, coloring_m, axis)
                if coloring_group > 1:
                    splits = Split(coloring_group, axis)(x)
                    outs = []
                    for i, split in enumerate(splits):
                        split_out = conditional_conv_layer(filters=m, number_of_classes=number_of_classes, name=name+str(i))([split, cls])
                        outs.append(split_out)
                    out = tf.keras.layers.Concatenate(axis)(outs)
                else:
                    out = conditional_conv_layer(filters=K.int_shape(x)[axis], number_of_classes=number_of_classes,
                                                 name=name)([x, cls])
                return out
            return f
    elif coloring == 'fconv':
        def after_norm_layer(axis, name):
            def f(x):
                coloring_group, m = get_m_group(x, coloring_m, axis)
                if coloring_group > 1:
                    splits = Split(coloring_group, axis)(x)
                    outs = []
                    for i, split in enumerate(splits):
                        split_out = factor_conv_layer(filters=m, number_of_classes=number_of_classes, name=name + '_c'+str(i), filters_emb=filters_emb, use_bias=False)([split, cls])
                        outs.append(split_out)
                    out = tf.keras.layers.Concatenate(axis)(outs)
                else:
                    out = factor_conv_layer(filters=K.int_shape(x)[axis], number_of_classes=number_of_classes, name=name + '_c', filters_emb=filters_emb, use_bias=False)([x, cls])
                return out
            return f
    elif coloring == 'uconv':
        def after_norm_layer(axis, name):
            def f(x):
                coloring_group, m = get_m_group(x, coloring_m, axis)
                if coloring_group > 1:
                    splits = Split(coloring_group, axis)(x)
                    outs = []
                    for i, split in enumerate(splits):
                        split_out = uncoditional_conv_layer(filters=m, kernel_size=(1, 1), name=name+str(i))(split)
                        outs.append(split_out)
                    out = tf.keras.layers.Concatenate(axis)(outs)
                else:
                    out = uncoditional_conv_layer(filters=K.int_shape(x)[axis], kernel_size=(1, 1), name=name)(x)
                return out
            return f
    elif coloring == 'ucconv':
        def after_norm_layer(axis, name):
            def f(x):
                coloring_group, m = get_m_group(x, coloring_m, axis)
                if coloring_group > 1:
                    splits = Split(coloring_group, axis)(x)
                    cs = []
                    us = []
                    for i, split in enumerate(splits):
                        split_c = conditional_conv_layer(filters=m, number_of_classes=number_of_classes, name=name + '_c'+str(i))([split, cls])
                        split_u = uncoditional_conv_layer(kernel_size=(1, 1), filters=K.int_shape(x)[axis], name=name + '_u'+str(i))(split)
                        cs.append(split_c)
                        us.append(split_u)
                    c = tf.keras.layers.Concatenate(axis)(cs)
                    u = tf.keras.layers.Concatenate(axis)(us)
                else:
                    c = conditional_conv_layer(filters=K.int_shape(x)[axis],
                                               number_of_classes=number_of_classes, name=name + '_c')([x, cls])
                    u = uncoditional_conv_layer(kernel_size=(1, 1), filters=K.int_shape(x)[axis], name=name + '_u')(x)
                out = Add(name=name + '_a')([c, u])
                return out
            return f
    elif coloring == 'ccsuconv':
        def after_norm_layer(axis, name):
            def f(x):
                coloring_group, m = get_m_group(x, coloring_m, axis)
                c = ConditionalCenterScale(number_of_classes=number_of_classes, axis=axis, name=name + '_c')([x, cls])
                if coloring_group > 1:
                    splits = Split(coloring_group, axis)(x)
                    us = []
                    for i, split in enumerate(splits):
                        split_u = uncoditional_conv_layer(kernel_size=(1, 1), filters=m, name=name + '_u'+str(i))(split)
                        us.append(split_u)
                    u = tf.keras.layers.Concatenate(axis)(us)
                else:
                    u = uncoditional_conv_layer(kernel_size=(1, 1), filters=K.int_shape(x)[axis], name=name + '_u')(x)
                out = Add(name=name + '_a')([c, u])
                return out
            return f
    elif coloring == 'ufconv':
        def after_norm_layer(axis, name):
            def f(x):
                coloring_group, m = get_m_group(x, coloring_m, axis)
                if coloring_group > 1:
                    splits = Split(coloring_group, axis)(x)
                    cs = []
                    us = []
                    for i, split in enumerate(splits):
                        split_c = factor_conv_layer(number_of_classes=number_of_classes, name=name + '_c'+str(i),
                                     filters=m, filters_emb=filters_emb,
                                     use_bias=False)([split, cls])
                        split_u = uncoditional_conv_layer(kernel_size=(1, 1), filters=m, name=name + '_u'+str(i))(split)
                        cs.append(split_c)
                        us.append(split_u)
                    c = tf.keras.layers.Concatenate(axis)(cs)
                    u = tf.keras.layers.Concatenate(axis)(us)
                else:
                    c = factor_conv_layer(number_of_classes=number_of_classes, name=name + '_c',
                                          filters=K.int_shape(x)[axis], filters_emb=filters_emb,
                                          use_bias=False)([x, cls])
                    u = uncoditional_conv_layer(kernel_size=(1, 1), filters=K.int_shape(x)[axis], name=name + '_u')(x)
                out = Add(name=name + '_a')([c, u])
                return out
            return f
    elif coloring == 'n':
        after_norm_layer = lambda axis, name: lambda x: x

    def result_norm(axis, name):
        def stack(inp):
            out = inp
            out = norm_layer(axis=axis, name=name + '_npart')(out)
            out = after_norm_layer(axis=axis, name=name + '_repart')(out)
            return out
        return stack

    return result_norm


def make_generator(input_noise_shape=(128,), output_channels=3, input_cls_shape=(1, ),
                   block_sizes=(128, 128, 128), resamples=("UP", "UP", "UP"),
                   first_block_shape=(4, 4, 128), number_of_classes=10, concat_cls=False,
                   block_norm='u', block_coloring='cs', filters_emb=10,
                   last_norm='u', last_coloring='cs',
                   decomposition='cholesky', whitten_m=0, coloring_m=0, iter_num=5, instance_norm=0, device='cpu',
                   gan_type=None, arch='res', spectral=False,
                   before_conv=0,
                   fully_diff_spectral=False, spectral_iterations=1, conv_singular=True,):

    assert arch in ['res', 'dcgan']
    inp = Input(input_noise_shape, name='GInputImage')
    cls = Input(input_cls_shape, dtype='int32', name='GLabel')

    if spectral:
        conv_layer = partial(SNConv2D, conv_singular=conv_singular,
                             fully_diff_spectral=fully_diff_spectral, spectral_iterations=spectral_iterations)
        cond_conv_layer = partial(SNConditionalConv11,
                                  fully_diff_spectral=fully_diff_spectral, spectral_iterations=spectral_iterations)
        dense_layer = partial(SNDense,
                              fully_diff_spectral=fully_diff_spectral, spectral_iterations=spectral_iterations)
        emb_layer = partial(SNEmbeding, fully_diff_spectral=fully_diff_spectral, spectral_iterations=spectral_iterations)
        factor_conv_layer = partial(SNFactorizedConv11,
                                    fully_diff_spectral=fully_diff_spectral, spectral_iterations=spectral_iterations)
    else:
        conv_layer = Conv2D
        cond_conv_layer = ConditionalConv11
        dense_layer = Dense
        emb_layer = Embedding
        factor_conv_layer = FactorizedConv11

    if concat_cls:
        y = emb_layer(input_dim=number_of_classes, output_dim=first_block_shape[-1])(cls)
        y = Reshape((first_block_shape[-1], ))(y)
        y = Concatenate(axis=-1)([y, inp])
    else:
        y = inp

    y = dense_layer(units=np.prod(first_block_shape), kernel_initializer=glorot_init)(y)
    y = Reshape(first_block_shape)(y)

    block_norm_layer = create_norm(block_norm, block_coloring,
                                   decomposition=decomposition,
                                   whitten_m=whitten_m, coloring_m=coloring_m,
                                   iter_num=iter_num, instance_norm=instance_norm, device=device,
                                   cls=cls, number_of_classes=number_of_classes, filters_emb=filters_emb,
                                   uncoditional_conv_layer=conv_layer, conditional_conv_layer=cond_conv_layer,
                                   factor_conv_layer=factor_conv_layer)

    last_norm_layer = create_norm(last_norm, last_coloring,
                                  decomposition=decomposition,
                                  whitten_m=whitten_m, coloring_m=coloring_m,
                                  iter_num=iter_num, instance_norm=instance_norm,  device=device,
                                  cls=cls, number_of_classes=number_of_classes, filters_emb=filters_emb,
                                  uncoditional_conv_layer=conv_layer, conditional_conv_layer=cond_conv_layer,
                                  factor_conv_layer=factor_conv_layer)

    i = 0
    for block_size, resample in zip(block_sizes, resamples):
        if arch == 'res':
            y = resblock(y, kernel_size=(3, 3), resample=resample,
                         nfilters=block_size, name='Generator.' + str(i),
                         norm=block_norm_layer, is_first=False, conv_layer=conv_layer)
        else:
            # TODO: SN DECONV
            y = dcblock(y, kernel_size=(4, 4), resample=resample,
                        nfilters=block_size, name='Generator.' + str(i),
                        norm=block_norm_layer, is_first=False, conv_layer=Conv2DTranspose, before_conv=before_conv)
        i += 1

    y = last_norm_layer(axis=-1, name='Generator.BN.Final')(y)
    y = Activation('relu')(y)
    output = conv_layer(filters=output_channels, kernel_size=(3, 3), name='Generator.Final',
                        kernel_initializer=glorot_init, use_bias=True, padding='same')(y)
    output = Activation('tanh')(output)

    if gan_type is None:
        return Model(inputs=[inp], outputs=output)
    else:
        return Model(inputs=[inp, cls], outputs=output)


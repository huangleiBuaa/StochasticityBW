import os
import sys
import time

sys.path.append(os.path.abspath('./gan'))

from gan.args import parser_with_default_args, get_generator_params

from generator import make_generator

import numpy as np


def speed(in_shape, output_channels, input_cls_shape,
          block_sizes, resamples, first_block_shape,
          number_of_classes, concat_cls,
          block_norm, block_coloring,
          last_norm, last_coloring,
          filters_emb,
          m, decom, d,
          iter_num, instance_norm,
          gan_type, arch, spectral, before_conv,
          fully_diff_spectral, spectral_iterations, conv_singular,
          batch_size=64, repeat=100):
    generator = make_generator(input_noise_shape=in_shape, output_channels=output_channels, input_cls_shape=input_cls_shape,
                               block_sizes=block_sizes, resamples=resamples, first_block_shape=first_block_shape,
                               number_of_classes=number_of_classes, concat_cls=concat_cls,
                               block_norm=block_norm, block_coloring=block_coloring, filters_emb=filters_emb,
                               last_norm=last_norm, last_coloring=last_coloring,
                               decomposition=decom, whitten_m=m, coloring_m=m,
                               iter_num=iter_num, instance_norm=instance_norm, device=d,
                               gan_type=gan_type, arch=arch, spectral=spectral,
                               before_conv=before_conv,
                               fully_diff_spectral=fully_diff_spectral,
                               spectral_iterations=spectral_iterations,
                               conv_singular=conv_singular,)
    # generator.summary()

    print()
    print('start test on {}:'.format(d))
    print('model:generator, arch:{}, blocks:{}, decom:{}'.format(arch, block_sizes, decom))
    inputs1 = np.random.normal(size=[1] + in_shape)
    inputs2 = np.random.normal(size=[batch_size] + in_shape)

    generator.predict(inputs1)  # warm up devices
    t1 = time.time()
    for _ in range(repeat):
        outputs = generator.predict(inputs2)
    t2 = time.time()
    item = ','.join(['device_' + d, 'm_' + str(m), 'decomposition_' + decom])
    print(item, ':', (t2 - t1) * 1000 / repeat)
    del generator


def main():
    args = parser_with_default_args()
    generator_params = get_generator_params(args)
    print()
    for device in ['cpu', 'gpu']:
        speed([128, ], generator_params.output_channels, generator_params.input_cls_shape,
              generator_params.block_sizes, generator_params.resamples, generator_params.first_block_shape,
              generator_params.number_of_classes, generator_params.concat_cls,
              generator_params.block_norm, generator_params.block_coloring,
              generator_params.last_norm, generator_params.last_coloring,
              generator_params.filters_emb,
              generator_params.whitten_m, generator_params.decomposition, device,
              generator_params.iter_num, generator_params.instance_norm,
              generator_params.gan_type, generator_params.arch, generator_params.spectral, generator_params.before_conv,
              generator_params.fully_diff_spectral, generator_params.spectral_iterations, generator_params.conv_singular,
              batch_size=64, repeat=100)


if __name__ == "__main__":
    main()

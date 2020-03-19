import os
import sys

sys.path.append(os.path.abspath('./gan'))


from tensorflow.python.keras.optimizers import Adam
from gan.dataset import LabeledArrayDataset
from gan.args import parser_with_default_args, get_generator_params, get_discriminator_params
from gan.train import Trainer
from archs.ac_gan import AC_GAN
from archs.projective_gan import ProjectiveGAN
from archs.gan import GAN

import json
from functools import partial
from scorer import compute_scores
from time import time

from generator import make_generator
from discriminator import make_discriminator

from tensorflow.python.keras import backend as K
import tensorflow as tf


def get_dataset(dataset, batch_size, supervised = False, noise_size=(128, )):
    from cifar10 import load_data
    (X, y), (X_test, y_test) = load_data()

    return LabeledArrayDataset(X=X, y=y if supervised else None, X_test=X_test, y_test=y_test,
                               batch_size=batch_size, noise_size=noise_size)


def compile_and_run(dataset, args, generator_params, discriminator_params):
    additional_info = json.dumps(vars(args))

    args.generator_optimizer = Adam(args.generator_lr, beta_1=args.beta1, beta_2=args.beta2)
    args.discriminator_optimizer = Adam(args.discriminator_lr, beta_1=args.beta1, beta_2=args.beta2)

    log_file = os.path.join(args.output_dir, 'log.txt')

    at_store_checkpoint_hook = partial(compute_scores, image_shape=args.image_shape, log_file=log_file,
                                       dataset=dataset, images_inception=args.samples_inception,
                                       images_fid=args.samples_fid, additional_info=additional_info,
                                       cache_file=args.fid_cache_file)

    lr_decay_schedule_generator, lr_decay_schedule_discriminator = get_lr_decay_schedule(args)

    generator_checkpoint = args.generator_checkpoint
    discriminator_checkpoint = args.discriminator_checkpoint

    generator = make_generator(**vars(generator_params))
    discriminator = make_discriminator(**vars(discriminator_params))

    generator.summary()
    discriminator.summary()

    if generator_checkpoint is not None:
        generator.load_weights(generator_checkpoint)

    if discriminator_checkpoint is not None:
        discriminator.load_weights(discriminator_checkpoint)

    hook = partial(at_store_checkpoint_hook, generator=generator)

    if args.phase == 'train':
        GANS = {None: GAN, 'AC_GAN': AC_GAN, 'PROJECTIVE': ProjectiveGAN}
        gan = GANS[args.gan_type](generator=generator, discriminator=discriminator,
                                  lr_decay_schedule_discriminator=lr_decay_schedule_discriminator,
                                  lr_decay_schedule_generator=lr_decay_schedule_generator,
                                  **vars(args))
        trainer = Trainer(dataset, gan, at_store_checkpoint_hook=hook, **vars(args))
        trainer.train()
    else:
        hook(0)


def get_lr_decay_schedule(args):
    number_of_iters_generator = 1000. * args.number_of_epochs
    number_of_iters_discriminator = 1000. * args.number_of_epochs * args.training_ratio

    if args.lr_decay_schedule is None:
        def lr_decay_schedule_generator(iter):
            return 1.

        def lr_decay_schedule_discriminator(iter):
            return 1.
    elif args.lr_decay_schedule == 'linear':
        def lr_decay_schedule_generator(iter):
            return tf.maximum(0., 1. - tf.cast(iter, 'float32') / number_of_iters_generator)

        def lr_decay_schedule_discriminator(iter):
            return tf.maximum(0., 1. - tf.cast(iter, 'float32') / number_of_iters_discriminator)
    elif args.lr_decay_schedule == 'half-linear':
        def lr_decay_schedule_generator(iter):
            return tf.where(tf.less(iter, tf.cast(number_of_iters_generator / 2, 'int64')),
                            tf.maximum(0., 1. - (tf.cast(iter, 'float32') / number_of_iters_generator)), 0.5)

        def lr_decay_schedule_discriminator(iter):
            return tf.where(tf.less(iter, tf.cast(number_of_iters_discriminator / 2, 'int64')),
                            tf.maximum(0., 1. - (tf.cast(iter, 'float32') / number_of_iters_discriminator)), 0.5)
    elif args.lr_decay_schedule == 'linear-end':
        decay_at = 0.828

        number_of_iters_until_decay_generator = number_of_iters_generator * decay_at
        number_of_iters_until_decay_discriminator = number_of_iters_discriminator * decay_at

        number_of_iters_after_decay_generator = number_of_iters_generator * (1 - decay_at)
        number_of_iters_after_decay_discriminator = number_of_iters_discriminator * (1 - decay_at)

        def lr_decay_schedule_generator(iter):
            return tf.where(tf.greater(iter, K.cast(number_of_iters_until_decay_generator, 'int64')),
                            tf.maximum(0., 1. - (K.cast(iter, 'float32') - number_of_iters_until_decay_generator) / number_of_iters_after_decay_generator), 1)

        def lr_decay_schedule_discriminator(iter):
            return tf.where(tf.greater(iter, K.cast(number_of_iters_until_decay_discriminator, 'int64')),
                            tf.maximum(0., 1. - (K.cast(iter, 'float32') - number_of_iters_until_decay_discriminator) / number_of_iters_after_decay_discriminator), 1)
    elif args.lr_decay_schedule.startswith("dropat"):
        drop_at = int(args.lr_decay_schedule.replace('dropat', ''))
        drop_at_generator = drop_at * 1000
        drop_at_discriminator = drop_at * 1000 * args.training_ratio
        print("Drop at generator %s" % drop_at_generator)

        def lr_decay_schedule_generator(iter):
            return tf.where(tf.less(iter, drop_at_generator), 1.,  0.1) * \
                   tf.maximum(0., 1. - tf.cast(iter, 'float32') / number_of_iters_generator)

        def lr_decay_schedule_discriminator(iter):
            return tf.where(tf.less(iter, drop_at_discriminator), 1.,  0.1) * \
                   tf.maximum(0., 1. - tf.cast(iter, 'float32') / number_of_iters_discriminator)
    else:
        assert False

    return lr_decay_schedule_generator, lr_decay_schedule_discriminator


def main():
    args = parser_with_default_args()
    print(args)
    dataset = get_dataset(dataset=args.dataset,
                          batch_size=args.batch_size,
                          supervised=args.gan_type is not None)

    args.output_dir = "output/%s_%s_%s" % (args.name, args.phase, time())
    # args.output_dir = "output/%s_%s" % (args.name, time())
    print(args.output_dir)
    args.checkpoints_dir = args.output_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(args.output_dir + '/config.json'):
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as outfile:
            json.dump(vars(args), outfile, indent=4)

    image_shape_dict = {'mnist': (28, 28, 1),
                        'fashion-mnist': (28, 28, 1),
                        'cifar10': (32, 32, 3),
                        'cifar100': (32, 32, 3),
                        'stl10': (48, 48, 3),
                        'imagenet': (128, 128, 3),
                        'tiny-imagenet': (64, 64, 3)}

    args.image_shape = image_shape_dict[args.dataset]
    print("Image shape %s x %s x %s" % args.image_shape)
    args.fid_cache_file = args.checkpoints_dir + "/%s_fid.npz" % args.dataset

    discriminator_params = get_discriminator_params(args)
    generator_params = get_generator_params(args)

    del args.dataset

    compile_and_run(dataset, args, generator_params, discriminator_params)


if __name__ == "__main__":
    main()

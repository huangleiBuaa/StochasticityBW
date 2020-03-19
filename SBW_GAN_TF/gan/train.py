import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as plt
from functools import partial
from tensorflow.python.keras import backend as K
assert K.image_data_format() == 'channels_last', "Backend should be tensorflow and data_format channel_last"
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)
from tqdm import tqdm


class Trainer(object):
    def __init__(self, dataset, gan, output_dir='output/generated_samples',
                 checkpoints_dir='output/checkpoints', training_ratio=5,
                 display_ratio=1, checkpoint_ratio=10, start_epoch=0,
                 number_of_epochs=100, batch_size=64, generator_batch_multiple=2,
                 at_store_checkpoint_hook=None, save_weights_only=True,
                 concatenate_generator_batches=True, **kwargs):
        self.dataset = dataset
        self.current_epoch = start_epoch
        self.last_epoch = start_epoch + number_of_epochs
        self.gan = gan
        self.gen_batch_mul = generator_batch_multiple
        self.concatenate_generator_batches = concatenate_generator_batches

        self.at_store_checkpoint_hook = at_store_checkpoint_hook
        self.save_weights_only = save_weights_only

        self.generator = gan.get_generator()
        self.discriminator = gan.get_discriminator()

        self.generator_train_op = gan.compile_generator_train_op()
        self.discriminator_train_op = gan.compile_discriminator_train_op()
        self.generate_op = gan.compile_generate_op()
        self.validate_op = gan.compile_validate_op()

        self.batch_size = batch_size
        self.output_dir = output_dir
        self.checkpoints_dir = checkpoints_dir
        self.training_ratio = training_ratio
        self.display_ratio = display_ratio
        self.checkpoint_ratio = checkpoint_ratio

    def save_generated_images(self):
        if hasattr(self.dataset, 'next_generator_sample_test'):
            batch = self.dataset.next_generator_sample_test()
        else:
            batch = self.dataset.next_generator_sample()
        gen_images = self.generate_op(batch + [False])
        image = self.dataset.display(gen_images, batch)
        title = "epoch_{}.png".format(str(self.current_epoch).zfill(3))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        plt.imsave(os.path.join(self.output_dir, title), image,  cmap='gray')

    def make_checkpoint(self):
        g_title = "epoch_{}_generator.h5".format(str(self.current_epoch).zfill(3))
        d_title = "epoch_{}_discriminator.h5".format(str(self.current_epoch).zfill(3))

        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        if self.save_weights_only:
            self.discriminator.save_weights(os.path.join(self.checkpoints_dir, d_title))
            self.generator.save_weights(os.path.join(self.checkpoints_dir, g_title))
        else:
            self.discriminator.save(os.path.join(self.checkpoints_dir, d_title))
            self.generator.save(os.path.join(self.checkpoints_dir, g_title))
 
        if self.at_store_checkpoint_hook is not None:
            self.at_store_checkpoint_hook(self.current_epoch)

    def train_one_step(self, discriminator_loss_list, generator_loss_list):
        for j in range(self.training_ratio):
            discriminator_batch = self.dataset.next_discriminator_sample()
            generator_batch = self.dataset.next_generator_sample()
            loss = self.discriminator_train_op(discriminator_batch + generator_batch + [True])
            discriminator_loss_list.append(loss)

        if self.concatenate_generator_batches:
            if self.gen_batch_mul != 1:
                generator_batch = []
                for i in range(self.gen_batch_mul):
                    generator_batch.append(self.dataset.next_generator_sample())
                generator_batch = [np.concatenate(l, axis=0) for l in zip(*generator_batch)]
            else:
                generator_batch = self.dataset.next_generator_sample()
            loss = self.generator_train_op(generator_batch + [True])
            generator_loss_list.append(loss)
        else:
            for j in range(self.gen_batch_mul):
                generator_batch = self.dataset.next_generator_sample()
                loss = self.generator_train_op(generator_batch + [True])
                generator_loss_list.append(loss)

    def train_one_epoch(self, validation_epoch=False):
        print("Epoch: %i" % self.current_epoch)
        discriminator_loss_list = []
        generator_loss_list = []

        for _ in tqdm(range(int(self.dataset.number_of_batches_per_epoch())), ascii=True):
            try:
                self.train_one_step(discriminator_loss_list, generator_loss_list)
            except tf.errors.InvalidArgumentError as err:
                print(err)

        g_loss_str, d_loss_str = self.gan.get_losses_as_string(np.mean(np.array(generator_loss_list), axis=0),
                                                               np.mean(np.array(discriminator_loss_list), axis=0),
                                                               tb_writer=self.tb_writer,
                                                               step=self.current_epoch)
        print(g_loss_str)
        print(d_loss_str)

        if hasattr(self.dataset, 'next_generator_sample_test') and validation_epoch:
            batches = int(self.dataset.number_of_batches_per_validation())
            if batches > 0:
                print("Validation...")
                validation_loss_list = []
                for _ in tqdm(range(int(self.dataset.number_of_batches_per_validation())), ascii=True):
                    generator_batch = self.dataset.next_generator_sample_test()
                    loss = self.validate_op(generator_batch + [True])
                    validation_loss_list.append(loss)
                val_loss_str, d_loss_str = self.gan.get_losses_as_string(np.mean(np.array(validation_loss_list), axis=0),
                                                                         np.mean(np.array(discriminator_loss_list), axis=0))
                print(val_loss_str.replace('Generator loss', 'Validation loss'))

        print("Discriminator lr %s" % K.get_value(self.gan.discriminator_optimizer.lr))
        print("Generator lr %s" % K.get_value(self.gan.generator_optimizer.lr))
        
    def train(self):
        sess = K.get_session()
        self.merged = tf.summary.merge_all()
        self.tb_writer = tf.summary.FileWriter(self.output_dir, sess.graph)
        self.at_store_checkpoint_hook = partial(self.at_store_checkpoint_hook, tb_writer=self.tb_writer, step=0)
        init = tf.global_variables_initializer()
        sess.run(init)
        while self.current_epoch < self.last_epoch:
            if (self.current_epoch + 1) % self.display_ratio == 0:
                self.save_generated_images()
            self.train_one_epoch((((self.current_epoch + 1) % self.checkpoint_ratio == 0) or self.current_epoch==0))
            if (self.current_epoch + 1) % self.checkpoint_ratio == 0:
                self.make_checkpoint()
            self.at_store_checkpoint_hook = partial(self.at_store_checkpoint_hook, step=self.current_epoch + 1)
            self.current_epoch += 1

        if (self.current_epoch + 1) % self.display_ratio == 0:
            self.save_generated_images()
        self.make_checkpoint()

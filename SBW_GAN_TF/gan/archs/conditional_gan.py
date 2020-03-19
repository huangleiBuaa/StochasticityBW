from tensorflow.python.keras.models import Model, Input
from tensorflow.python.keras.layers import Dense, Flatten, Concatenate, Activation, Dropout
from tensorflow.python.keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D, Cropping2D
from tensorflow.python.keras.layers.normalization import InstanceNormalization
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
import tensorflow.python.keras.backend as K

from gan.dataset import UGANDataset
from gan.args import parser_with_default_args
from gan.train import Trainer
from archs.ac_gan import AC_GAN
from gan.layers.normalization import ConditionalInstanceNormalization

import numpy as np
from skimage.transform import resize
from skimage.io import imread
import os

from itertools import chain
from sklearn.utils import shuffle
from scipy.ndimage.morphology import distance_transform_edt
from skimage.color import gray2rgb


def block(out, nkernels, down=True, bn=True, dropout=False, leaky=True, normalization=InstanceNormalization):
    if leaky:
        out = LeakyReLU(0.2) (out)
    else:
        out = Activation('relu')(out)
    if down:
        out = ZeroPadding2D((1, 1)) (out)
        out = Conv2D(nkernels, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(out)
    else:
        out = Conv2DTranspose(nkernels, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(out)
        out = Cropping2D((1,1))(out)
    if bn:
        out = normalization(axis=-1)(out)
    if dropout:
        out = Dropout(0.5)(out)
    return out


def make_generator(image_size, number_of_classes):
    input_a = Input(image_size + (1,))
    cls = Input((1, ), dtype='int32')
    # input is 64 x 64 x nc
    conditional_instance_norm = lambda axis: (lambda inp: ConditionalInstanceNormalization(number_of_classes=number_of_classes, axis=axis)([inp, cls]))

    e1 = ZeroPadding2D((1, 1))(input_a)
    e1 = Conv2D(64, kernel_size=(4, 4), strides=(2, 2))(e1)
    #input is 32 x 32 x 64
    e2 = block(e1, 128, normalization=conditional_instance_norm)
    #input is 16 x 16 x 128
    e3 = block(e2, 256, normalization=conditional_instance_norm)
    #input is 8 x 8 x 256
    e4 = block(e3, 512, normalization=conditional_instance_norm)
    #input is 4 x 4 x 512
    e5 = block(e4, 512, normalization=conditional_instance_norm)
    #input is 2 x 2 x 512
    e6 = block(e5, 512, bn = False)
    #input is 1 x 1 x 512
    out = block(e6, 512, down=False, leaky=False, dropout=True, normalization=conditional_instance_norm)
    #input is 2 x 2 x 512
    out = Concatenate(axis=-1)([out, e5])
    out = block(out, 512, down=False, leaky=False, dropout=True, normalization=conditional_instance_norm)
    #input is 4 x 4 x 512
    out = Concatenate(axis=-1)([out, e4])
    out = block(out, 512, down=False, leaky=False, dropout=True, normalization=conditional_instance_norm)
    #input is 8 x 8 x 512
    out = Concatenate(axis=-1)([out, e3])
    out = block(out, 512, down=False, leaky=False, normalization=conditional_instance_norm)
    #input is 16 x 16 x 512
    out = Concatenate(axis=-1)([out, e2])
    out = block(out, 256, down=False, leaky=False, normalization=conditional_instance_norm)
    #input is 32 x 32 x 256
    out = Concatenate(axis=-1)([out, e1])
    out = block(out, 3, down=False, leaky=False, bn=False)
    #input is  64 x 64 x 128

    out = Activation('tanh')(out)

    return Model(inputs=[input_a, cls], outputs=[out])


def make_discriminator(image_size, number_of_classes):
    input_a = Input(image_size + (3,))
    input_b = Input(image_size + (1,))
    out = Concatenate(axis=-1)([input_a, input_b])
    out = Conv2D(64, kernel_size=(4, 4), strides=(2, 2))(out)
    out = block(out, 128)
    out = block(out, 256)
    real_vs_fake = block(out, 1, bn=False)
    real_vs_fake = Flatten()(real_vs_fake)
    cls = Flatten()(out)
    cls = Dense(128, activation='relu')(cls)
    cls = Dense(number_of_classes)(cls)

    return Model(inputs=[input_a, input_b], outputs=[real_vs_fake, cls])


class CGAN(AC_GAN):
    def __init__(self, image_size, l1_weigh_penalty=100,  **kwargs):
        self.gt_image_placeholder = Input(image_size + (3, ))
        super(CGAN, self).__init__(**kwargs)
        self.l1_weight_penalty = l1_weigh_penalty
        self.additional_inputs_for_generator_train = [self.gt_image_placeholder]

    def compile_intermediate_variables(self):
        self.generator_output = [self.generator(self.generator_input), self.generator_input[0]]
        self.discriminator_fake_output = self.discriminator(self.generator_output)
        self.discriminator_real_output = self.discriminator(self.discriminator_input)

    def additional_generator_losses(self):
        loss_list = super(CGAN, self).additional_generator_losses()
        l1_loss = self.l1_weight_penalty * K.mean(K.abs(self.gt_image_placeholder -
                                                        self.generator_output[0]))
        loss_list.append(l1_loss)
        self.generator_metric_names.append('l1')
        return loss_list

class SketchDataset(UGANDataset):
    def __init__(self, images_folder, sketch_folder, batch_size, invalid_images_files, test_set, number_of_classes, image_size):
        super(SketchDataset, self).__init__(batch_size, None)

        self.images_folder = images_folder
        self.sketch_folder = sketch_folder
        self.invalid_images_files = invalid_images_files
        self.test_set = test_set
        self.image_size = image_size
        self.number_of_classes = number_of_classes

        self.load_names()

        self._batches_before_shuffle = len(self.images_train) / self._batch_size
        self.test_data_index = 0

    def number_of_batches_per_validation(self):
        return len(self.images_test) / self._batch_size

    def next_generator_sample_test(self):
        index = np.arange(self.test_data_index, self.test_data_index + self._batch_size)
        index = index % self.images_test.shape[0]
        test_data = self._load_data_batch(index, stage='test')
        self.test_data_index += self._batch_size
        return list(test_data)

    def load_names(self):
        class_names = sorted(os.listdir(self.sketch_folder))[:self.number_of_classes]
        self.class_label_dict = dict(zip(class_names, range(len(class_names))))

        invalid_sketches = set(chain(*[open(f).read().split('\n') for f in self.invalid_images_files]))
        test_images = {t.split('/')[1].split('.')[0] for t in open(self.test_set).read().split('\n') if t}

        self.images_train = []
        self.images_test = []

        self.sketch_train = []
        self.sketch_test = []

        self.labels_train = []
        self.labels_test = []

        for class_name in self.class_label_dict.keys():
            for sketch_name in os.listdir(os.path.join(self.sketch_folder, class_name)):
                name = sketch_name.split('-')[0]

                if name in invalid_sketches:
                    continue
                image = os.path.join(self.images_folder, class_name, name + '.jpg')
                sketch = os.path.join(self.sketch_folder, class_name, sketch_name)
                label = self.class_label_dict[class_name]

                if name not in test_images:
                    self.images_train.append(image)
                    self.sketch_train.append(sketch)
                    self.labels_train.append(label)
                else:
                    self.labels_test.append(label)
                    self.sketch_test.append(sketch)
                    self.images_test.append(image)

        self.images_train = np.array(self.images_train)
        self.images_test = np.array(self.images_test)

        self.sketch_train = np.array(self.sketch_train)
        self.sketch_test = np.array(self.sketch_test)

        self.labels_train = np.array(self.labels_train)
        self.labels_test = np.array(self.labels_test)

    def _load_data_batch(self, index, stage='train'):
        load_from_folder = lambda names: [resize(imread(name), self.image_size, preserve_range=True)
                                          for name in names[index]]

        if stage == 'train':
            sketches = load_from_folder(self.sketch_train)
            labels = self.labels_train[index]
            images = load_from_folder(self.images_train)
        else:
            sketches = load_from_folder(self.sketch_test)
            labels = self.labels_test[index]
            images = load_from_folder(self.images_test)

        labels = np.expand_dims(labels, axis=1)
        sketches = np.array([distance_transform_edt(np.mean(sketch, axis=-1)>254)[..., np.newaxis] for sketch in sketches])
        images = self.preprocess_image(np.array(images))

        return sketches, labels, images
    
    def preprocess_image(self, image):
        image /= 255
        image -= 0.5
        image *= 2
        return image
    
    def deprocess(self, image):
        image /= 2
        image += 0.5
        image *= 255
        return image.astype(np.uint8)
        
    def next_generator_sample(self):
        index = self._next_data_index()
        sketches, labels, images = self._load_data_batch(index)
        return [sketches, labels, images]

    def next_discriminator_sample(self):
        index = self._next_data_index()
        sketches, labels, images = self._load_data_batch(index)
        return [images, sketches, labels]

    def _shuffle_data(self):
        self.sketch_train, self.images_train, self.labels_train = shuffle(self.sketch_train, self.images_train, self.labels_train)
        
    def display(self, output_batch, input_batch=None):
        gen_images = output_batch[0]
        sketches = input_batch[0]
        gen_images = super(SketchDataset, self).display(gen_images)
        sketches = super(SketchDataset, self).display(sketches)
        #Transform distance field to rbg image
        sketches = np.squeeze(sketches)
        sketches /= sketches.max()
        sketches = gray2rgb(sketches)
        sketches = 2 * (sketches - 0.5)

        result = self.deprocess(np.concatenate([sketches, gen_images], axis=1))
        return result


def main():
    parser = parser_with_default_args()
    parser.add_argument("--images_folder", default="data/photo/tx_000100000000", help='Folder with photos')
    parser.add_argument("--sketch_folder", default="data/sketch/tx_000000000000", help='Folder with sketches')
    parser.add_argument("--invalid_files", default= ['data/info/invalid-ambiguous.txt', 'data/info/invalid-context.txt',
                                                     'data/info/invalid-error.txt', 'data/info/invalid-pose.txt'],
                        help='List of files with invalid sketches, comma separated', type=lambda x: x.split(','))
    parser.add_argument("--test_set", default='data/info/testset.txt', help='File with test set')
    parser.add_argument("--image_size", default=(64, 64), help='Size of the images')
    parser.add_argument("--number_of_classes", default=2, help='Number of classes to train on, usefull for debugging')
    parser.add_argument("--cache_dir", default='tmp', help='Store distance transforms to this folder.')

    args = parser.parse_args()

    dataset = SketchDataset(images_folder=args.images_folder,
                            sketch_folder=args.sketch_folder,
                            batch_size=args.batch_size,
                            invalid_images_files=args.invalid_files,
                            test_set=args.test_set,
                            number_of_classes=args.number_of_classes,
                            image_size=args.image_size)

    generator = make_generator(image_size=args.image_size, number_of_classes=args.number_of_classes)
    discriminator = make_discriminator(image_size=args.image_size, number_of_classes=args.number_of_classes)
    generator.summary()
    discriminator.summary()

    gan = CGAN(generator=generator, discriminator=discriminator, **vars(args))
    trainer = Trainer(dataset, gan, **vars(args))

    trainer.train()


if __name__ == "__main__":
    main()

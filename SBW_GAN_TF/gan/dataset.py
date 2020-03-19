import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import pylab as plt
from skimage.transform import resize
from skimage import img_as_ubyte
from sklearn.utils import shuffle


class UGANDataset(object):
    def __init__(self, batch_size, noise_size):
        self._batch_size = batch_size
        self._noise_size = noise_size
        self._batches_before_shuffle = 0
        self._current_batch = 0

    def number_of_batches_per_epoch(self):
        return self._batches_before_shuffle

    def next_generator_sample(self):
        return [np.random.normal(size=(self._batch_size,) + self._noise_size)]
    
    def _load_discriminator_data(self, index):
        assert False, "Should be implimented in subclasses"

    def _shuffle_data(self):
        assert False, "Should be implimented in subclasses"

    def _next_data_index(self):
        self._current_batch %= self._batches_before_shuffle
        if self._current_batch == 0:
            self._shuffle_data()
        index = np.arange(self._current_batch * self._batch_size, (self._current_batch + 1) * self._batch_size)
        self._current_batch += 1
        return index

    def next_discriminator_sample(self):
        index = self._next_data_index()
        image_batch = self._load_discriminator_data(index)
        return image_batch        

    def display(self, output_batch, input_batch=None, row=None, col=None):
        row = output_batch.shape[0] if row is None else row
        col = 1 if col is None else col
        batch = output_batch
        height, width = batch.shape[1], batch.shape[2]
        total_width, total_height = width * col, height * row
        result_image = np.empty((total_height, total_width, batch.shape[3]), dtype=output_batch.dtype)
        batch_index = 0
        for i in range(row):
            for j in range(col):
                result_image[(i * height):((i+1)*height), (j * width):((j+1)*width)] = batch[batch_index]
                batch_index += 1
        return result_image

    
class ArrayDataset(UGANDataset):
    def __init__(self, X, batch_size, noise_size):
        super(ArrayDataset, self).__init__(batch_size, noise_size)
        self._X = X
        self._batches_before_shuffle = X.shape[0] // self._batch_size + 1
    
    def _load_discriminator_data(self, index):
        index = index % self._X.shape[0]
        return [self._X[index]]
    
    def _shuffle_data(self):
        np.random.shuffle(self._X)


class FolderDataset(UGANDataset):
    def __init__(self, input_dir, batch_size, noise_size, image_size):
        super(FolderDataset, self).__init__(batch_size, noise_size)        
        self._image_names = np.array(os.listdir(input_dir))
        self._input_dir = input_dir
        self._image_size = image_size
        self._batches_before_shuffle = self._image_names.shape[0] // self._batch_size + 1      
        
    def _preprocess_image(self, img):
        return resize(img, self._image_size) * 2 - 1
    
    def _deprocess_image(self, img):
        return img_as_ubyte((img + 1) / 2)
        
    def _load_discriminator_data(self, index):
        index = index % len(self._image_names)
        return [np.array([self._preprocess_image(plt.imread(os.path.join(self._input_dir, img_name)))
                          for img_name in self._image_names[index]])]
    
    def _shuffle_data(self):
        np.random.shuffle(self._image_names)
        
    def display(self, output_batch, input_batch = None):
        image = super(FolderDataset, self).display(output_batch)
        return self._deprocess_image(image)


class LabeledArrayDataset(ArrayDataset):
    def __init__(self, X, X_test, batch_size, y=None, y_test=None, noise_size=(128, ), dequantize = True):
        X = (X.astype(np.float32) - 127.5) / 127.5
        X_test = (X_test.astype(np.float32) - 127.5) / 127.5
        
        if dequantize:
            X += np.random.uniform(0, 1/128.0, size=X.shape)
        super(LabeledArrayDataset, self).__init__(X, batch_size, noise_size)

        self._Y = y
        self._Y_test = y_test
        self._X_test = X_test
        if y is not None:
            if len(y.shape) == 1:
                self._Y = np.expand_dims(y, axis=1)
            self._cls_prob = np.bincount(np.squeeze(self._Y, axis=1)) / float(self._Y.shape[0])
            self.number_of_classes = len(np.unique(self._Y))

        if y_test is not None:
            if len(y_test.shape) == 1:
                self._Y_test = np.expand_dims(y_test, axis=1)

    def number_of_batches_per_epoch(self):
        return 1000

    def number_of_batches_per_validation(self):
        return 0

    def next_generator_sample(self):
        labels = [] if self._Y is None else self.current_discriminator_labels
        return [np.random.normal(size=(self._batch_size,) + self._noise_size)] + labels

    def next_generator_sample_test(self):
        labels = [] if self._Y is None else [np.random.randint(self.number_of_classes, size=(self._batch_size, 1))]
        # [(np.arange(self._batch_size) % self.number_of_classes).reshape((self._batch_size,1))]
        return [np.random.normal(size=(self._batch_size,) + self._noise_size)] + labels

    def _load_discriminator_data(self, index):
        index = index % self._X.shape[0]
        if self._Y is not None:
            self.current_discriminator_labels = [self._Y[index]]
        else:
            self.current_discriminator_labels = []
        return [self._X[index]] + self.current_discriminator_labels

    def _shuffle_data(self):
        x_shape = self._X.shape
        self._X = self._X.reshape((x_shape[0], -1))
        if self._Y is None:
            self._X = shuffle(self._X)
        else:
            self._X, self._Y = shuffle(self._X, self._Y)
        self._X = self._X.reshape(x_shape)

    def display(self, output_batch, input_batch=None):
        batch = output_batch[0]
        image = super(LabeledArrayDataset, self).display(batch)
        image = (image * 127.5) + 127.5
        image = np.squeeze(np.round(image).astype(np.uint8))
        return image

    
    


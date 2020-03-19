from metrics.inception_score import get_inception_score
from metrics.fid import calculate_fid_given_arrays
from gan.dataset import UGANDataset
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras.backend as K
from skimage.io import imsave
import os
import pickle


def draw_grid(fname, images, labels=None,  nrows=10, ncols=10):
    if labels is None:
        sample_images = images[:nrows * ncols]
    else:
        sample_images = []
        for cls_index in range(ncols):
            sample_images.append(images[labels == cls_index][:nrows])
        sample_images = np.concatenate(sample_images, axis=0)
    sample_images = sample_images.astype('uint8')
    image = UGANDataset(None, None).display(sample_images, None, nrows, ncols)
    imsave(fname, image)


def save_images(dir_name, images, labels):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    #Load class metadata
    f = open('synset_words.txt')
    name_to_synset = {}
    for line in f.readlines():
        name, synset = line.split(' ', 1)
        name_to_synset[name] = synset[:-1]
    #print (name_to_synset)
    with open('ti_classses.pkl') as f:
        index_to_name = pickle.load(f)

    for i in np.unique(labels):
        name = name_to_synset[index_to_name[int(i)]]
        sample_images = images[labels == i][:16].astype('uint8')
        sample_images = sample_images.astype('uint8')
        image = UGANDataset(None, None).display(sample_images, None, 4, 4)
        imsave(os.path.join(dir_name, name + '.jpg'), image)


def compute_scores(epoch, image_shape, generator, dataset, images_inception=50000, images_fid=10000,
                   log_file=None, cache_file='mnist_fid.npz', additional_info="", tb_writer=None, step=0):
    compute_inception = images_inception != 0
    compute_fid = images_fid != 0
    number_of_images = max(images_inception, images_fid)

    if not (compute_inception or compute_fid):
        return
    images = np.empty((number_of_images, ) + image_shape)
    labels = np.empty((number_of_images, ))
    generator_input = generator.get_input_at(0)
    if type(generator_input) != list:
        generator_input = [generator_input]

    predict_fn = K.function(generator_input + [K.learning_phase()], [generator.get_output_at(0)])
    
    bs = dataset._batch_size
    conditional = False
    for begin in tqdm(range(0, number_of_images, bs), ascii=True):
        
        end = min(number_of_images, begin + bs)
        n_images = end - begin
        g_s = dataset.next_generator_sample_test()
        if len(g_s) == 2:
           labels[begin:end] = np.squeeze(g_s[1], axis=1)[:n_images]
           conditional = True
        images[begin:end] = predict_fn(g_s + [False])[0][:n_images]

    images *= 127.5
    images += 127.5

    def to_rgb(array):
        if array.shape[-1] != 3:
            #hack for grayscale mnist
            return np.concatenate([array, array, array], axis=-1)
        else:
            return array
    #save_images('baseline_16', to_rgb(images), labels)
    draw_grid(os.path.join(os.path.dirname(log_file), "epoch_%s_imagegrid.png" % epoch),
              to_rgb(images), labels if conditional else None)

    if compute_inception:
        mean, std = get_inception_score(to_rgb(images[:images_inception]))
        str = "INCEPTION SCORE: %s, %s" % (mean, std)
        print(str)
        if tb_writer and isinstance(tb_writer, tf.summary.FileWriter):
            summary = tf.Summary()
            summary.value.add(tag='IS', simple_value=mean)
            tb_writer.add_summary(summary, step)
        if log_file is not None:
            with open(log_file, 'a') as f:
                print(("Epoch %s " % (epoch, )) + str, file=f) #+ " " + additional_info

    if compute_fid:
        true_images = 127.5 * dataset._X_test + 127.5
        fid = calculate_fid_given_arrays([to_rgb(true_images)[:images_fid],
                                                            to_rgb(images)[:images_fid]], cache_file=cache_file)
        str = "FID SCORE: %s" % fid
        print(str)
        if tb_writer and isinstance(tb_writer, tf.summary.FileWriter):
            summary = tf.Summary()
            summary.value.add(tag='FID', simple_value=fid)
            tb_writer.add_summary(summary, step)
        if log_file is not None:
            with open(log_file, 'a') as f:
                print(("Epoch %s " % (epoch, )) + str, file=f) #+ " " + additional_info

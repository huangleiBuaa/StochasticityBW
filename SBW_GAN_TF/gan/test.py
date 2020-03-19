from tensorflow.python.keras import backend as K
from tqdm import tqdm
import numpy as np
import tensorflow as tf

assert K.image_data_format() == 'channels_last', "Backend should be tensorflow and data_format channel_last"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
K.set_session(session)


def deprocess_image(img):
    return (255 * ((img + 1) / 2.0)).astype(np.uint8)


def generate_images(dataset, generator,  number_of_samples, out_index=-1, deprocess_fn=deprocess_image):
    result = []

    for _ in tqdm(range(number_of_samples)):
        out = generator.predict(dataset.next_generator_sample())
        result.append(deprocess_fn(out[out_index]))
    result_array = np.concatenate(result, axis=0)
    del result
    return result_array

import numpy as np
from keras import backend as K
from keras.applications import vgg19
from keras.preprocessing.image import load_img, img_to_array

def preprocess_image(path, dim):
    img = load_img(path, target_size=dim)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x, h, w):
    if K.image_dim_ordering() == 'th':
        x = x.reshape((3, h, w))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((h, w, 3))

    x[:, :, 0] = x[:, :, 0] - 103.939
    x[:, :, 1] = x[:, :, 1] - 116.779
    x[:, :, 2] = x[:, :, 2] - 123.68

    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x
import keras

import tensorflow as tf
from skimage.io import imread,imsave
from skimage.color import rgb2gray,gray2rgb,rgb2lab,lab2rgb
from keras.models import Sequential
from keras.layers import Conv2D,UpSampling1D,InputLayer,Conv2DTranspose,UpSampling2D
from keras.preprocessing.image import img_to_array,load_img
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os

def image_color_test():
    # this method to let's know something about RGB and LAB color space
    test_image_gray = rgb2gray(imread('black.jpg'))
    test_image_rgb = imread('black.jpg')

    print(test_image_gray.shape)
    print(test_image_gray)
    print(test_image_rgb.shape)

def get_train_data(img_file):
    image=img_to_array(load_img(img_file))
    image_shape=image.shape
    image=np.array(image,dtype=float)
    x=rgb2lab(1.0 / 255 * image)[:,:,0]
    y=rgb2lab(1.0 / 255 * image)[:, :, 1:]
    x=x.reshape(1,image_shape[0],image_shape[1],1)
    y=y.reshape(1,image_shape[0],image_shape[1],2)
    return x,y,image_shape
def build_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(None, None, 1)))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
    model.compile(optimizer='rmsprop', loss='mse')
    return model
def train():
    x,y,img_shape=get_train_data('black.jpeg')
    model=build_model()
    num_epochs=1000
    batch_size=6
    model_file='simple_model.h5'
    model.fit(x,y,batch_size=1,epochs=1000)
    model.save(model_file)
if __name__ == '__main__':
    train()
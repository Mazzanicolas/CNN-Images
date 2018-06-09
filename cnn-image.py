#from keras.layers.core import Dense, Dropout, Activation, Flatten
#from keras.layers.convolutional import Convolution2D, MaxPooling2D
#from keras.optimizers import SGD, RMSprop, adam
#from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
#import theano
from PIL import Image
from numpy import *

#from sklearn.utils import shuffle
#from sklearn.cross_validation import train_test_split

#Full path to avoid permission denied
input_path  = 'C:/Users/cclar_000/Desktop/AI/CNN-MNIST-mod/input_data'
output_path = './output_data'

def prepareDataset(input_path, output_path, output_image_height, output_image_width,
                   labels_ammount_array):
    input_images = getImages(input_path)
    for file in input_images:
        image = input_path + '/' + file
        resized_image    = resizeImage(output_image_height, output_image_width, image)
        grayscaled_image = convertToGrayscale(resized_image)
        grayscaled_image.save(output_path + '\\' + file, 'JPEG')
    images_as_array = imagesToArray(output_path)
    labels = createLabels(labels_ammount_array)
    permutation_order = np.random.permutation(len(images_as_array))
    data, label = images_as_array[permutation_order], labels[permutation_order]
    train_data  = [data, label]
    return train_data
        

def resizeImage(height, width, image):
    input_image  = Image.open(image)
    return input_image.resize((height, width))

def convertToGrayscale(image):
    return image.convert('L')

def getImages(path):
    return os.listdir(path)

def imagesToArray(path):
    input_images = getImages(path)
    image_matrix = []
    for image in input_images:
        image_as_array = array(Image.open(path + '\\' + image)).flatten()
        image_matrix.append(image_as_array)
    return array(image_matrix, 'f')

#You can edit this method to use custom labels
def createLabels(labels_amount_array):
    labels = np.ones(np.sum(labels_amount_array),dtype = int)
    previous_label_begin = 0
    for index, label_amount in enumerate(labels_amount_array):
        next_label_begin = previous_label_begin + label_amount
        labels[previous_label_begin : next_label_begin] = index
        previous_label_begin = next_label_begin
    return labels

train_data = prepareDataset(input_path, output_path, 100, 100,[1,3,1])


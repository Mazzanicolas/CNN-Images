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
from random import randrange

#from sklearn.utils import shufflem #You can use this instead of shuffleData
#from sklearn.cross_validation import train_test_split #You can use this instead of splitData

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
    train_data = shuffleData(images_as_array, labels)
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

def shuffleData(data, labels):
    permutation_order = np.random.permutation(len(data))
    shuffled_data, shuffled_labels = data[permutation_order], labels[permutation_order]
    shuffled_data_labels  = [shuffled_data, shuffled_labels]
    return shuffled_data_labels

def splitData(dataset, split):    
    train_size = split * len(dataset)
    data   = dataset[0]
    labels = dataset[1]
    train_data   = list()
    train_labels = list()
    data_copy    = list(data)
    labels_copy  = list(labels)
    while len(train_data) < train_size:
        index = randrange(len(data_copy))
        train_data.append(data_copy.pop(index))
        train_labels.append(labels_copy.pop(index))
    return train_data, data_copy, train_labels, labels_copy

#Add parameter cmap = 'gray' in imshow for grayscale
def previewImage(images_as_array):
    img = images_as_array.reshape(image_height, image_width)
    plt.imshow(img)
    plt.show()

#Config
input_path  = './input_data'
output_path = './output_data'

image_height = 100
image_width  = 100

batch_size = 50
number_of_classes = 3
number_of_epoch   = 20

image_channels    = 1
number_of_filters = 32 #?
number_if_pool    = 2
number_of_convolution = 3

    
dataset = prepareDataset(input_path, output_path, image_height, image_width,[1,3,1])

X_train, X_test, y_train, y_test = splitData(dataset,0.5)

from keras.layers.core import Dense, Dropout, Activation, Flatten	
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils
from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator

import keras

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *

#Hack
from keras import backend as be
be.set_image_dim_ordering('th')


#Methods
def prepareData(input_path, output_path, output_image_height, output_image_width,
                   labels_amount_array):
    input_images = getImages(input_path)
    for file in input_images:
        image = input_path + '\\' + file
        resized_image    = resizeImage(output_image_height, output_image_width, image)
        grayscaled_image = convertToGrayscale(resized_image)
        grayscaled_image.save(output_path + '\\' + file, 'JPEG')
    images_as_array = imagesToArray(output_path)  
    labels = createLabels(labels_amount_array)
    data, labels = shuffle(images_as_array, labels, random_state=2)
    return data, labels
        
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
        open_image = Image.open(path + '\\' + image)
        image_as_array = imageToArray(open_image)
        image_matrix.append(image_as_array)
    return array(image_matrix, 'f')

def imageToArray(image):
    return array(image).flatten()
    

#You can edit this method to use custom labels
def createLabels(labels_amount_array):
    labels = np.ones(np.sum(labels_amount_array),dtype = int)
    previous_label_begin = 0
    for index, label_amount in enumerate(labels_amount_array):
        next_label_begin = previous_label_begin + label_amount
        labels[previous_label_begin : next_label_begin] = index
        previous_label_begin = next_label_begin
    return labels

#Add parameter cmap = 'gray' in imshow for grayscale
def previewImage(images_as_array):
    img = images_as_array.reshape(image_height, image_width)
    plt.imshow(img)
    plt.show()

def declareDimensionDepth(X_train, X_test):
    X_train = X_train.reshape(X_train.shape[0], image_depth, image_height, image_width)
    X_test  = X_test.reshape(X_test.shape[0], image_depth, image_height, image_width)
    return X_train, X_test

def convertDataType(X_train, X_test):
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    return X_train, X_test

def normalizeValues(X_train, X_test, value):
    X_train /= value
    X_test /= value
    return X_train, X_test

#Config
input_path  = './input_data'
output_path = './output_data'
model_name  = 'trained_model.h5'

if not os.path.exists(input_path):
    os.makedirs(input_path)
if not os.path.exists(output_path):
    os.makedirs(input_path)
    
image_height = 28
image_width  = 28
labels_amount = [100,# 300 samples of 0 class
                 100,# 300 samples of 1 class
                 100,# 300 samples of 2 class
                 100,# 300 samples of 3 class
                 100,# 300 samples of 4 class
                 100,# 300 samples of 5 class
                 100,# 300 samples of 6 class
                 100,# 300 samples of 7 class
                 100,# 300 samples of 8 class
                 100,# 300 samples of 9 class
                 100,# 300 samples of + class
                 100]# 300 samples of - class

batchs_size = 10
number_of_classes = 12
number_of_epoch   = 20

image_depth       = 1
number_of_filters = 32
number_if_pool    = 2
number_of_convolution = 3

activation_function = "relu"
output_nodes = 12

#Data set up
print('Preparing Data ...')
data, labels = prepareData(input_path, output_path, image_height, image_width, labels_amount)
print('Splitting Data ...')
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=4)
print('Adapting Data ...')
X_train, X_test = declareDimensionDepth(X_train, X_test)
X_train, X_test = convertDataType(X_train, X_test)
X_train, X_test = normalizeValues(X_train,X_test,255)

Y_train = np_utils.to_categorical(y_train, len(labels_amount))
Y_test  = np_utils.to_categorical(y_test, len(labels_amount))

#Model
def create_model():
    print('Creating model ...')
    model  = Sequential()
    model.add(Conv2D(number_of_filters, (number_of_convolution,
                            number_of_convolution), activation=activation_function,
                            input_shape=(image_depth,image_height,image_width)))

    model.add(Conv2D(number_of_filters, (number_of_convolution, number_of_convolution), activation=activation_function))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_nodes, activation='softmax'))
    model.save('trained_model.h5')
    return model

def compile_model(model):
    print('Compiling Model ...')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def train(model):
    print('Training Model ...')
    save_epoch = keras.callbacks.ModelCheckpoint('model_checkpoint.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    model.fit(X_train, Y_train, 
              batch_size=batchs_size, epochs=number_of_epoch, verbose=1, validation_data=(X_test, Y_test), callbacks=[save_epoch])
    return model

def start():
    model_name = 'model_checkpoint.hdf5' #Path
    if os.path.isfile(model_name):
        print('Existing model detected. Loading data ...')
        model = load_model(model_name)
    else:
        print('No model detected. Creating model ...')
        model = create_model()
    compiled_model = compile_model(model)
    trained_model= train(compiled_model)
    trained_model.save(model_name)

def predict(image_path):
    if os.path.isfile(model_name):
        print('Existing model detected. Loading data ...')
        model = load_model(model_name)
    else:
        print('No model detected. Create and train a new model')
        return
    image_generator = ImageDataGenerator().flow_from_directory(image_path, class_mode=None, target_size=(image_height, image_width), seed=10)
    print(image_generator)
    prediction = model.predict(image_generator)    
    print(prediction)

start()

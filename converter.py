from keras.models import load_model
from keras.layers import Input, Dense
from tensorflow import Tensor
from keras import backend as K
from keras.engine import InputLayer

model = load_model('model_checkpoint.hdf5')

for layer in model.layers:
	print(layer)


input_layer1 = InputLayer(input_shape=(51, 68, 3), name="input_1")
input_layer2 = InputLayer(input_shape=(51, 68, 3), name="input_2")
print ("input shape:", input_layer1.input_shape)
print ("input tensor:", input_layer1.input)
print ("name:", input_layer1.name)
print ("sparse:", input_layer1.sparse)
print ("dtype:", input_layer1.dtype)

model.layers[0] = input_layer1
model.layers[1] = input_layer2
model.save("reshaped-model.h5")

import coremltools
coreml_model = coremltools.converters.keras.convert('reshaped-model.h5', is_bgr=True, 	    
						    input_names=['image1', 'image2'], image_input_names=['image1', 'image2'],
						    output_names=['output'],
                                                    blue_bias=-103.939, green_bias=-116.779, red_bias=-123.68)


coreml_model.save('Output.mlmodel')

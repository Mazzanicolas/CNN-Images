
# 🔨 Work in progress 🔨
![alt text](https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png)

## Dependencies

### Python Libraries
You can use `pip install ` here.

| Library        | Version           |
| -------- |:-------:|
| Keras | 2.2.0 |
| Sklearn | 0.0 | 
| Numpy | 1.14.4 | 
| Matplotlib| 2.2.2 | 
| Theano | 1.0.2 | 
| Pillow | 5.1.0 | 

### Anaconda (Optional)

You might want to install [Anaconda](https://www.anaconda.com/download/).
Usually g++ is not available in theano. 
If you get
`WARNING (theano.configdefaults): g++ not available`
for better performance you can use `conda install m2w64-toolchain`

## Configuration
`input_path` folder for input images

`output_path` folder in whitch the converted images are saved

`image_height` output image height

`image_width` output image width 

`labels_amount` array with the amount of samples for image

`batchs_size` amount of samples per iteration

`number_of_classes` amount of classes in the dataset

`number_of_epoch`   amount of iterations in the dataset

`image_depth`  dimension for the depth of the input image

`activation_function` [activation function](https://keras.io/activations/)

`output_nodes` output corresponding to the amount of classes


## Usage
In this example we're going to make an extended version of the MNIST.
### Training the model
1. First fill your input folder with your images.
2. Adjust the parameters.
3. Navigate to the folder where the file `cnn-image.py` is located.
4. Run `python cnn-image.py`

### Evaluate model on test data
## Documentation

**prepareData** ( *input_path*, *output_path*, *output_image_height*, *output_image_width*, *labels_amount_array* ) **→ returns data, labels**

`Resizes dataset to a given heigth*width, convert in to grayscale, saves them in to a given folder ...`

input_path : *path to input dataset*.

output_path: *path to output dataset (modified data)*.

output_image_height: *heigth of the output image to train the model*.

output_image_width: *width of the output image to train the model*.

labels_amount_array: *array with size equal to classes and each index contains the amount of images of that class to trian the model **(images and labels must be in order)***.

**resizeImage** ( *height*, *width*, *image* ) **→ returns image**

`resize an image to a given heigth*width`

height : *height of the output image*.

width: *width of the output image*.

image : *image to convert*.

**convertToGrayscale** ( image ) **→ returns image**

`converts an image in to grayscale`

image : *image to convert*.

**getImages** ( path ) **→ returns images in path**

`returns all images in a given path`

path : *path to a folder with images*.

**imagesToArray** ( path ) **→ returns matrix**

`Converts multiple images into a matrix`

**imageToArray** ( image ) **→ returns array**

`Converts the image into an array and applies a flatten function to the array`

**createLabels** ( labels_amount_array ) **→ returns array**

`Creates an array of labels to assign to your data`

**previewImage** ( images_as_array ) **→ returns None**

`Displays the image with matplotlib`


**declareDimensionDepth** ( X_train, X_test ) 

**convertDataType** ( X_train, X_test )

**normalizeValues** ( X_train, X_test, value )

## Posible errors

`ModuleNotFoundError: No module named 'tensorflow'`

Keras has __tensorflow__ as default, we need to change it to __theano__. You can use tensorflow but you will have to make some changes.  

**Fix**: edit your `keras.json` in `C:\users\USERNAME\.keras\`


#### Use this configuration:

	 {    
	    "floatx": "float32",
	    "epsilon": 1e-07,
	    "backend": "theano",
	    "image_data_format": "channels_last"
	 }
## References
[Anuj shah](https://www.youtube.com/watch?v=2pQOXjpO_u0)

[Elite Data Science](https://elitedatascience.com/keras-tutorial-deep-learning-in-python)

[Keras](keras.io/)

[Scikit Learn](http://scikit-learn.org)

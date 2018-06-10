# 🔨 Work in progress 🔨
![alt text](https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png)

## Dependencies

You can use `pip install ` here.

| Library        | Version           |
| -------- |:-------:|
| Keras | 2.2.0 |
| Sklearn | 0.0 | 
| Numpy | 1.14.4 | 
| Matplotlib| 2.2.2 | 
| Theano | 1.0.2 | 
| Pillow | 5.1.0 | 

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

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

## Usage


## Documentation


## Posible errors


`ModuleNotFoundError: No module named 'tensorflow'`, edit your `keras.json` in `C:\users\USERNAME\.keras\`


#### Use this configuration:

    
	 {    
	    "floatx": "float32",
	    "epsilon": 1e-07,
	    "backend": "theano",
	    "image_data_format": "channels_last"
	 }

### Dependencies



* keras

* sklearn

* numpy

* matplotlib

* theano

* pillow

## Errors


`ModuleNotFoundError: No module named 'tensorflow'`, edit your keras.json in 'C:\users\USERNAME\.keras\'


#### Use this configuration:

    
{
    
    "floatx": "float32",

    "epsilon": 1e-07,

    "backend": "theano",

    "image_data_format": "channels_last"

}
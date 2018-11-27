import keras
from keras.layers.core import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.regularizers import *


def get_model():
	aliases = {}
	Input_1 = Input(shape=(1, 28, 28), name='Input_1')
	Convolution2D_3 = Convolution2D(name='Convolution2D_3',nb_row= 3,nb_col= 3,activation= 'relu' ,nb_filter= 32)(Input_1)
	MaxPooling2D_2 = MaxPooling2D(name='MaxPooling2D_2')(Convolution2D_3)
	Dropout_9 = Dropout(name='Dropout_9',p= 0.25)(MaxPooling2D_2)
	Convolution2D_4 = Convolution2D(name='Convolution2D_4',nb_row= 3,nb_col= 3,activation= 'relu' ,nb_filter= 64)(Dropout_9)
	Dropout_10 = Dropout(name='Dropout_10',p= 0.25)(Convolution2D_4)
	Convolution2D_5 = Convolution2D(name='Convolution2D_5',nb_row= 3,nb_col= 3,activation= 'relu' ,nb_filter= 128)(Dropout_10)
	Dropout_11 = Dropout(name='Dropout_11',p= 0.4)(Convolution2D_5)
	Flatten_4 = Flatten(name='Flatten_4')(Dropout_11)
	Dense_12 = Dense(name='Dense_12',output_dim= 128,activation= 'relu' )(Flatten_4)
	Dropout_12 = Dropout(name='Dropout_12',p= 0.40)(Dense_12)
	Dense_13 = Dense(name='Dense_13',output_dim= 10,activation= 'softmax' )(Dropout_12)

	model = Model([Input_1],[Dense_13])
	return aliases, model


from keras.optimizers import *

def get_optimizer():
	return Adadelta()

def is_custom_loss_function():
	return False

def get_loss_function():
	return 'categorical_crossentropy'

def get_batch_size():
	return 512

def get_num_epoch():
	return 100

def get_data_config():
	return '{"mapping": {"Image": {"options": {"Height": 28, "width_shift_range": 0, "vertical_flip": false,
	 "Resize": false, "pretrained": "None", "shear_range": 0, "rotation_range": 0, "height_shift_range": 0, 
	 "Augmentation": false, "horizontal_flip": false, "Normalization": false, "Scaling": 1, "Width": 28}, 
	 "port": "InputPort0", "shape": "", "type": "Image"}, "label": {"options": {}, "port": "OutputPort0", 
	 "shape": "", "type": "Categorical"}}, "numPorts": 1, "samples": {"training": 56000, "validation": 7000, 
	 "split": 4, "test": 7000}, "dataset": {"type": "private", "samples": 70000, "name": "Fashion_MNIST"},
	  "kfold": 1, "datasetLoadOption": "full", "shuffle": false}'
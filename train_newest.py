import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings(action='ignore')
import librosa
from sklearn.utils import shuffle
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
import tensorflow as tf

# Global vars
RANDOM_SEED = 1337
SAMPLE_RATE = 32000
SIGNAL_LENGTH = 4.0 # seconds
SPEC_SHAPE = (129,501)#height X width
FMIN = 500
FMAX = 12500
#MAX_AUDIO_FILES = 15000
from focal_loss import BinaryFocalLoss
import soundfile as sf

import tensorflow_addons as tfa
from sklearn.utils import shuffle
from PIL import Image
from tqdm import tqdm,tnrange,tqdm_notebook
import tensorflow as tf
from tqdm.keras import TqdmCallback
from keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import applications as app
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten,AveragePooling2D
from tensorflow.keras.layers import Dense,BatchNormalization,Dropout 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.applications import EfficientNetB4, ResNet50,ResNet101, VGG16, MobileNet, InceptionV3
from keras.regularizers import l2
# Make sure your experiments are reproducible
from tensorflow.keras.layers import Conv2D, Input, LeakyReLU, Flatten, Dense, Reshape, Conv2DTranspose,BatchNormalization, Activation
from tensorflow.keras import Model, Sequential
from tensorflow.keras import Model, Sequential
tf.random.set_seed(RANDOM_SEED)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random
import pandas as pd
from  ast import literal_eval
from keras import utils as np_utils
# Build a simple model as a sequence of  convolutional blocks.
# Each block has the sequence CONV --> RELU --> BNORM --> MAXPOOL.
# Finally, perform global average pooling and add 2 dense layers.
# The last layer is our classification layer and is softmax activated.
# (Well it's a multi-label task so sigmoid might actually be a better choice)

class SumLayer(tf.keras.layers.Layer):
	def __init__(self,**kwargs):
		super(SumLayer, self).__init__(**kwargs)
	#def get_config(self):
	#	config = super().get_config().copy()
	#	return 
	#def get_config(self):
		# Implement get_config to enable serialization. This is optional.
	#	base_config = super(SumLayer, self).get_config()
	#	config = {"initializer": tf.keras.initializers.serialize(self.initializer)}
	#	return dict(list(base_config.items()) + list(config.items()))

	def call(self, inputs):
		row_sum = tf.reduce_sum(inputs,2,keepdims=True)
		return row_sum
class SumLayer2(tf.keras.layers.Layer):
	def __init__(self,**kwargs):
		super(SumLayer2, self).__init__(**kwargs)
	#def get_config(self):
	#	config = super().get_config().copy()
	#	return 
	#def get_config(self):
		# Implement get_config to enable serialization. This is optional.
	#	base_config = super(SumLayer, self).get_config()
	#	config = {"initializer": tf.keras.initializers.serialize(self.initializer)}
	#	return dict(list(base_config.items()) + list(config.items()))

	def call(self, inputs):
		row_sum = tf.reduce_sum(inputs,2,keepdims=True)
		return row_sum
class SumLayer3(tf.keras.layers.Layer):
	def __init__(self,**kwargs):
		super(SumLayer3, self).__init__(**kwargs)
	#def get_config(self):
	#	config = super().get_config().copy()
	#	return 
	#def get_config(self):
		# Implement get_config to enable serialization. This is optional.
	#	base_config = super(SumLayer, self).get_config()
	#	config = {"initializer": tf.keras.initializers.serialize(self.initializer)}
	#	return dict(list(base_config.items()) + list(config.items()))

	def call(self, inputs):
		row_sum = tf.reduce_sum(inputs,2,keepdims=True)
		return row_sum

class ScaleLayer(tf.keras.layers.Layer):
	def __init__(self,**kwargs):
		super(ScaleLayer, self).__init__(**kwargs)
	#def get_config(self):
	#	config = super().get_config().copy()
	#	return config
	#def get_config(self):
        # Implement get_config to enable serialization. This is optional.
	#	base_config = super(ScaleLayer, self).get_config()
		#config = {"initializer": keras.initializers.serialize(self.initializer)}
	#	return dict(list(base_config.items()) + list(config.items()))

	def call(self, inputs):
		row_sum = tf.reduce_sum(inputs,2,keepdims=True) + 10**-8 #added this to stop nans
		input_norm = tf.divide(inputs,row_sum)
		return input_norm

class ScaleLayer2(tf.keras.layers.Layer):
	def __init__(self,**kwargs):
		super(ScaleLayer2, self).__init__(**kwargs)
	#def get_config(self):
	#	config = super().get_config().copy()
	#	return config
	#def get_config(self):
        # Implement get_config to enable serialization. This is optional.
	#	base_config = super(ScaleLayer, self).get_config()
		#config = {"initializer": keras.initializers.serialize(self.initializer)}
	#	return dict(list(base_config.items()) + list(config.items()))

	def call(self, inputs):
		row_sum = tf.reduce_sum(inputs,2,keepdims=True) + 10**-8 #added this to stop nans
		input_norm = tf.divide(inputs,row_sum)
		return input_norm

class ScaleLayer3(tf.keras.layers.Layer):
	def __init__(self,**kwargs):
		super(ScaleLayer3, self).__init__(**kwargs)
	#def get_config(self):
	#	config = super().get_config().copy()
	#	return config
	#def get_config(self):
        # Implement get_config to enable serialization. This is optional.
	#	base_config = super(ScaleLayer, self).get_config()
		#config = {"initializer": keras.initializers.serialize(self.initializer)}
	#	return dict(list(base_config.items()) + list(config.items()))

	def call(self, inputs):
		row_sum = tf.reduce_sum(inputs,2,keepdims=True) + 10**-8 #added this to stop nans
		input_norm = tf.divide(inputs,row_sum)
		return input_norm


def model_tf(num_classes,l2_str=10**-4):
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3),input_shape=(None, None, 1),strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.Conv2D(16, (3, 3),strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.MaxPooling2D((2, 2)),

# Second conv block
    tf.keras.layers.Conv2D(32, (3, 3),strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.Conv2D(32, (3, 3),strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.MaxPooling2D((2, 2)), 

# Third conv block        
    tf.keras.layers.Conv2D(64, (3, 3),strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.Conv2D(64, (3, 3),strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.MaxPooling2D((2, 2)), 
        
        
  
# Fourth conv block
    tf.keras.layers.Conv2D(128, (3, 3),strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.Conv2D(128, (3, 3),strides=1, padding='same'),
    tf.keras.layers.ReLU(),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D((2, 2)), 
        
# Fifth conv block
    tf.keras.layers.Conv2D(512, (3, 3),strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.Conv2D(512, (3, 3),strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.MaxPooling2D((2, 2)), 
        
        
# Sixth conv block
    tf.keras.layers.Conv2D(512, (3, 3),strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),  
    tf.keras.layers.ReLU(),

    tf.keras.layers.Conv2D(512, (3, 3),strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.MaxPooling2D((2, 2)), 
        
 # Seventh conv block
    tf.keras.layers.Conv2D(1024, (2, 2),strides=1, padding='valid'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),          
 # Eighth conv block
    tf.keras.layers.Conv2D(2048, (1, 1)),#activation='sigmoid'),
    tf.keras.layers.BatchNormalization(), 
    tf.keras.layers.ReLU(),
 #Final BLock
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(2048, activation='relu',kernel_regularizer=l2(l2_str), bias_regularizer=l2(l2_str)),   
    #tf.keras.layers.Dropout(0.5),  
    tf.keras.layers.Dense(1024, activation='relu',kernel_regularizer=l2(l2_str), bias_regularizer=l2(l2_str)),
    #tf.keras.layers.Dropout(0.5),   
    tf.keras.layers.Dense(512, activation='relu',kernel_regularizer=l2(l2_str), bias_regularizer=l2(l2_str)),   
    #tf.keras.layers.Dropout(0.5),])
    tf.keras.layers.Dense(num_classes, activation='sigmoid')])
    ## a fully covolutional NN !
    print('MODEL HAS {} PARAMETERS.'.format(model.count_params()))
    return model


def model_mult_att_CNN(num_classes,l2_str=0): ## four att blocks 2nd , 4th and 6th and one already
	inputs = Input(shape=[128,None,1])
	x = inputs
	#first conv block
	x = Conv2D(filters=16, kernel_size=[3,3], strides=1, padding='same',use_bias=False,
				kernel_initializer=tf.keras.initializers.HeNormal(),)(x)
	x = BatchNormalization()(x)
	x = tf.keras.layers.ReLU()(x)

	x = Conv2D(filters=16, kernel_size=[3,3], strides=1, padding='same',use_bias=False,
				kernel_initializer=tf.keras.initializers.HeNormal())(x)
	x = BatchNormalization()(x)
	x = tf.keras.layers.ReLU()(x)
	x = tf.keras.layers.MaxPooling2D((2, 2))(x)


	# Second conv block
	x= Conv2D(32, (3, 3),strides=1, padding='same',use_bias=False,
				kernel_initializer=tf.keras.initializers.HeNormal())(x)
	x = BatchNormalization()(x)
	x = tf.keras.layers.ReLU()(x)
		
	x = tf.keras.layers.Conv2D(32, (3, 3),strides=1, padding='same',use_bias=False,
				kernel_initializer=tf.keras.initializers.HeNormal())(x)
	x = tf.keras.layers.ReLU()(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.MaxPooling2D((2, 2))(x) 

	skip2 = x
	# Third conv block
		
		
		
	x = tf.keras.layers.Conv2D(64, (3, 3),strides=1, padding='same',use_bias=False,
				kernel_initializer=tf.keras.initializers.HeNormal())(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.ReLU()(x)

	x = tf.keras.layers.Conv2D(64, (3, 3),strides=1, padding='same',use_bias=False,
				kernel_initializer=tf.keras.initializers.HeNormal())(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.ReLU()(x)
	x = tf.keras.layers.MaxPooling2D((2, 2))(x) 
		
		

	# Fourth conv block
		
	x = tf.keras.layers.Conv2D(128, (3, 3),strides=1, padding='same',use_bias=False,
				kernel_initializer=tf.keras.initializers.HeNormal())(x)
	x =tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.ReLU()(x)
	x = tf.keras.layers.Conv2D(128, (3, 3),strides=1, padding='same',use_bias=False,
				kernel_initializer=tf.keras.initializers.HeNormal())(x)

	x= tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.ReLU()(x)
	x =tf.keras.layers.MaxPooling2D((2, 2))(x) 
	skip4 = x
	# Fifth conv block
	x = tf.keras.layers.Conv2D(512, (3, 3),strides=1, padding='same',use_bias=False,
				kernel_initializer=tf.keras.initializers.HeNormal())(x)
	x =tf.keras.layers.BatchNormalization()(x)
	x =tf.keras.layers.ReLU()(x)
	x = tf.keras.layers.Conv2D(512, (3, 3),strides=1, padding='same',use_bias=False,
				kernel_initializer=tf.keras.initializers.HeNormal())(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x =tf.keras.layers.ReLU()(x)
		
	x = tf.keras.layers.MaxPooling2D((2, 2))(x)
		
		
	# Sixth conv block
	x =tf.keras.layers.Conv2D(512, (3, 3),strides=1, padding='same',use_bias=False,
				kernel_initializer=tf.keras.initializers.HeNormal())(x)
	x =tf.keras.layers.BatchNormalization()(x)  
	x =tf.keras.layers.ReLU()(x)
	x =tf.keras.layers.Conv2D(512, (3, 3),strides=1, padding='same',use_bias=False,
				kernel_initializer=tf.keras.initializers.HeNormal())(x)
	x=tf.keras.layers.BatchNormalization()(x)
	x=tf.keras.layers.ReLU()(x)
	x= tf.keras.layers.MaxPooling2D((2, 2))(x) 
	skip6 = x  
	# Seventh conv block
	x=tf.keras.layers.Conv2D(1024, (2, 2),strides=1, padding='valid',use_bias=False,
				kernel_initializer=tf.keras.initializers.HeNormal())(x)
	x=tf.keras.layers.BatchNormalization()(x)
	x=tf.keras.layers.ReLU()(x)
	# Global pooling instead of flatten()
	##att2
	##att2 = tf.keras.layers.Conv2D(num_classes, (1, 1),use_bias=False,
	##			kernel_initializer=tf.keras.initializers.HeNormal())(skip2)#,activation='softmax'
	##att2=tf.keras.layers.BatchNormalization()(att2)
	##att2=tf.keras.layers.Softmax()(att2)
	##att2 = tf.keras.layers.Lambda(ScaleLayer())(att2)
	##z2 = tf.keras.layers.Conv2D(num_classes, (1, 1),use_bias=False,
	##			kernel_initializer=tf.keras.initializers.HeNormal())(skip2) #,activation='sigmoid'
	#3z2 = tf.keras.layers.BatchNormalization()(z2)
	##z2 = tf.keras.layers.Activation('sigmoid')(z2)
	##att2_op = tf.keras.layers.multiply([att2,z2])
	##att2_op= tf.keras.layers.Lambda(SumLayer())(att2_op)
	##att2_op = tf.keras.layers.GlobalAveragePooling2D()(att2_op)
	##att4
	#att4 = tf.keras.layers.Conv2D(num_classes, (1, 1),use_bias=False,
	#			kernel_initializer=tf.keras.initializers.HeNormal())(skip4)#,activation='softmax'
	#att4=tf.keras.layers.BatchNormalization()(att4)
	#att4=tf.keras.layers.Softmax()(att4)
	#att4 = ScaleLayer()(att4)
	#z4 = tf.keras.layers.Conv2D(num_classes, (1, 1),use_bias=False,
	#			kernel_initializer=tf.keras.initializers.HeNormal())(skip4) #,activation='sigmoid'
	#z4 = tf.keras.layers.BatchNormalization()(z4)
	#z4 = tf.keras.layers.Activation('sigmoid')(z4)
	#att4_op = tf.keras.layers.multiply([att4,z4])
	#att4_op= SumLayer()(att4_op)
	#att4_op = tf.keras.layers.GlobalAveragePooling2D()(att4_op)
	##att6
	att6 = tf.keras.layers.Conv2D(num_classes, (1, 1),use_bias=False,
				kernel_initializer=tf.keras.initializers.HeNormal())(skip6)#,activation='softmax'
	att6=tf.keras.layers.BatchNormalization()(att6)
	att6=tf.keras.layers.Softmax()(att6)
	att6 = ScaleLayer2()(att6)
	z6 = tf.keras.layers.Conv2D(num_classes, (1, 1),use_bias=False,
				kernel_initializer=tf.keras.initializers.HeNormal())(skip6) #,activation='sigmoid'
	z6 = tf.keras.layers.BatchNormalization()(z6)
	z6 = tf.keras.layers.Activation('sigmoid')(z6)
	att6_op = tf.keras.layers.multiply([att6,z6])
	att6_op= SumLayer2()(att6_op)
	att6_op = tf.keras.layers.GlobalAveragePooling2D()(att6_op)
	# Eighth conv block
	y=tf.keras.layers.Conv2D(num_classes, (1, 1),use_bias=False,
				kernel_initializer=tf.keras.initializers.HeNormal())(x)

	y=tf.keras.layers.BatchNormalization()(y)
	y=tf.keras.layers.Softmax()(y)
	y = ScaleLayer()(y)
	z = tf.keras.layers.Conv2D(num_classes, (1, 1),use_bias=False,
				kernel_initializer=tf.keras.initializers.HeNormal())(x) #,activation='sigmoid'
	z = tf.keras.layers.BatchNormalization()(z)
	z = tf.keras.layers.Activation('sigmoid')(z)
	w = tf.keras.layers.multiply([y,z])
	w = SumLayer()(w)
	w  = tf.keras.layers.Flatten()(w)
	op = tf.keras.layers.concatenate([att6_op,w],axis=-1)
	#op = tf.keras.layers.Flatten()(op)
	#op = tf.keras.layers.Dense(397,activation='relu')(op)
	op = tf.keras.layers.Dense(397,activation='sigmoid')(op)
	#tf.keras.layers.GlobalAveragePooling2D()]) 
		#a fully covolutional NN !
	model = Model(inputs=inputs, outputs=op)
	print('MODEL HAS {} PARAMETERS.'.format(model.count_params()))
	return model

def model_tf_multiatt(num_classes):
	inputs = Input(shape=[128,625,1])
	x = inputs
	x = tf.keras.layers.Conv2D(16, (3, 3),strides=1, padding='same')(x)

	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.ReLU()(x)
	x = tf.keras.layers.Conv2D(16, (3, 3),strides=1, padding='same')(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.ReLU()(x)
	x = tf.keras.layers.MaxPooling2D((2, 2))(x)

	# Second conv block
	x = tf.keras.layers.Conv2D(32, (3, 3),strides=1, padding='same')(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.ReLU()(x)
		
	x = tf.keras.layers.Conv2D(32, (3, 3),strides=1, padding='same')(x)
	x = tf.keras.layers.ReLU()(x)
	x =tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.MaxPooling2D((2, 2))(x)

	# Third conv block
		
		
		
	x = tf.keras.layers.Conv2D(64, (3, 3),strides=1, padding='same')(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.ReLU()(x)

	x = tf.keras.layers.Conv2D(64, (3, 3),strides=1, padding='same')(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.ReLU()(x)
	x = tf.keras.layers.MaxPooling2D((2, 2))(x)
		
		

	# Fourth conv block
	x =tf.keras.layers.Conv2D(128, (3, 3),strides=1, padding='same')(x)
	x =tf.keras.layers.BatchNormalization()(x)
	x =tf.keras.layers.ReLU()(x)

	x =tf.keras.layers.Conv2D(128, (3, 3),strides=1, padding='same')(x)
	x =tf.keras.layers.BatchNormalization()(x)
	x =tf.keras.layers.ReLU()(x)
	x =tf.keras.layers.MaxPooling2D((2, 2))(x)
		
	# Fifth conv block
	x =tf.keras.layers.Conv2D(512, (3, 3),strides=1, padding='same')(x)
	x =tf.keras.layers.BatchNormalization()(x)
	x =tf.keras.layers.ReLU()(x)
	x =tf.keras.layers.Conv2D(512, (3, 3),strides=1, padding='same')(x)
	x =tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.ReLU()(x)
	x = tf.keras.layers.MaxPooling2D((2, 2))(x)
	skip5 = x
	# Sixth conv block
	x =tf.keras.layers.Conv2D(512, (3, 3),strides=1, padding='same')(x)
	x =tf.keras.layers.BatchNormalization()(x)
	x =tf.keras.layers.ReLU()(x)
	x =tf.keras.layers.Conv2D(512, (3, 3),strides=1, padding='same')(x)
	x =tf.keras.layers.BatchNormalization()(x)
	x =tf.keras.layers.ReLU()(x)
	x =tf.keras.layers.MaxPooling2D((2, 2))(x)
	skip6 = x
		# Seventh conv block
	x =tf.keras.layers.Conv2D(1024, (2, 2),strides=1, padding='valid')(x)
	x =tf.keras.layers.BatchNormalization()(x)
	x =tf.keras.layers.ReLU()(x)
	skip7 = x
	# Global pooling instead of flatten()
		# Eighth conv block
	x =tf.keras.layers.Conv2D(num_classes, (1, 1))(x)
	x =tf.keras.layers.BatchNormalization()(x)
	x =tf.keras.layers.ReLU()(x)
	x =tf.keras.layers.GlobalAveragePooling2D()(x)
	###dense layers
	#x =tf.keras.layers.Dense(2048,activation='relu')(x)
	att5 = tf.keras.layers.Conv2D(num_classes, (1, 1))(skip5)
	att5 =tf.keras.layers.BatchNormalization()(att5)
	att5 =tf.keras.layers.ReLU()(att5)
	att5 =tf.keras.layers.GlobalAveragePooling2D()(att5)
	##att6:
	att6 = tf.keras.layers.Conv2D(num_classes, (1, 1))(skip6)
	att6 =tf.keras.layers.BatchNormalization()(att6)
	att6 =tf.keras.layers.ReLU()(att6)
	att6 =tf.keras.layers.GlobalAveragePooling2D()(att6)
	# att6 =tf.keras.layers.Dense(2048,activation='relu')(att6)
	##att7:
	att7 = tf.keras.layers.Conv2D(num_classes, (1, 1))(skip7)
	att7 =tf.keras.layers.BatchNormalization()(att7)
	att7 =tf.keras.layers.ReLU()(att7)
	att7 =tf.keras.layers.GlobalAveragePooling2D()(att7)
	# att7 =tf.keras.layers.Dense(2048,activation='relu')(att7)
	##concatenate here
	op = tf.keras.layers.concatenate([att5,att6,att7,x],axis=-1)
	op = tf.keras.layers.Flatten()(op)
	#op = tf.keras.layers.Dense(2048,activation="relu")(op)
	op = tf.keras.layers.Dense(512,activation="relu")(op)
	op = tf.keras.layers.Dense(num_classes,activation='sigmoid')(op)
	model = Model(inputs=inputs, outputs=op)
	return model

#########################################################################################################################################################
def model_tf_att(num_classes,l2_strength=0,l2_str=0):
	inputs = Input(shape=[128,625,1])
	x = inputs
    	#first conv block
	x = Conv2D(filters=16, kernel_size=[3,3], strides=1, padding='same',use_bias=False,
              kernel_initializer=tf.keras.initializers.HeNormal(),
              kernel_regularizer=tf.keras.regularizers.l2(l2_strength), bias_regularizer=l2(l2_str))(x)
	x = BatchNormalization()(x)
	x = tf.keras.layers.ReLU()(x)
	x = Conv2D(filters=16, kernel_size=[3,3], strides=1, padding='same',use_bias=False,
              kernel_initializer=tf.keras.initializers.HeNormal(),
              kernel_regularizer=tf.keras.regularizers.l2(l2_strength), bias_regularizer=l2(l2_str))(x)
	x = BatchNormalization()(x)
	x = tf.keras.layers.ReLU()(x)
	x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    

	# Second conv block
	x= Conv2D(32, (3, 3),strides=1, padding='same',use_bias=False,
              kernel_initializer=tf.keras.initializers.HeNormal(),
              kernel_regularizer=tf.keras.regularizers.l2(l2_strength), bias_regularizer=l2(l2_str))(x)
	x = BatchNormalization()(x)
	x = tf.keras.layers.ReLU()(x)
	x = tf.keras.layers.Conv2D(32, (3, 3),strides=1, padding='same',use_bias=False,
              kernel_initializer=tf.keras.initializers.HeNormal(),
              kernel_regularizer=tf.keras.regularizers.l2(l2_strength), bias_regularizer=l2(l2_str))(x)
	x = tf.keras.layers.ReLU()(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.MaxPooling2D((2, 2))(x) 

	# Third conv block
        
        
	x = tf.keras.layers.Conv2D(64, (3, 3),strides=1, padding='same',use_bias=False,
              kernel_initializer=tf.keras.initializers.HeNormal(),
              kernel_regularizer=tf.keras.regularizers.l2(l2_strength), bias_regularizer=l2(l2_str))(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.ReLU()(x)
	x = tf.keras.layers.Conv2D(64, (3, 3),strides=1, padding='same',use_bias=False,
              kernel_initializer=tf.keras.initializers.HeNormal(),
              kernel_regularizer=tf.keras.regularizers.l2(l2_strength), bias_regularizer=l2(l2_str))(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.ReLU()(x)
	x = tf.keras.layers.MaxPooling2D((2, 2))(x) 
        
        
  
  	  # Fourth conv block
        
	x = tf.keras.layers.Conv2D(128, (3, 3),strides=1, padding='same',use_bias=False,
              kernel_initializer=tf.keras.initializers.HeNormal(),
              kernel_regularizer=tf.keras.regularizers.l2(l2_strength), bias_regularizer=l2(l2_str))(x)
	x =tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.ReLU()(x)
	x = tf.keras.layers.Conv2D(128, (3, 3),strides=1, padding='same',use_bias=False,
              kernel_initializer=tf.keras.initializers.HeNormal(),
              kernel_regularizer=tf.keras.regularizers.l2(l2_strength), bias_regularizer=l2(l2_str))(x)
	x= tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.ReLU()(x)
	x =tf.keras.layers.MaxPooling2D((2, 2))(x) 
        
	# Fifth conv block
	x = tf.keras.layers.Conv2D(512, (3, 3),strides=1, padding='same',use_bias=False,
              kernel_initializer=tf.keras.initializers.HeNormal(),
              kernel_regularizer=tf.keras.regularizers.l2(l2_strength), bias_regularizer=l2(l2_str))(x)
	x =tf.keras.layers.BatchNormalization()(x)
	x =tf.keras.layers.ReLU()(x)
	x = tf.keras.layers.Conv2D(512, (3, 3),strides=1, padding='same',use_bias=False,
              kernel_initializer=tf.keras.initializers.HeNormal(),
              kernel_regularizer=tf.keras.regularizers.l2(l2_strength), bias_regularizer=l2(l2_str))(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x =tf.keras.layers.ReLU()(x)
	x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        
        
	# Sixth conv block
	x =tf.keras.layers.Conv2D(512, (3, 3),strides=1, padding='same',use_bias=False,
              kernel_initializer=tf.keras.initializers.HeNormal(),
              kernel_regularizer=tf.keras.regularizers.l2(l2_strength), bias_regularizer=l2(l2_str))(x)
	x =tf.keras.layers.BatchNormalization()(x)  
	x =tf.keras.layers.ReLU()(x)
	x =tf.keras.layers.Conv2D(512, (3, 3),strides=1, padding='same',use_bias=False,
              kernel_initializer=tf.keras.initializers.HeNormal(),
              kernel_regularizer=tf.keras.regularizers.l2(l2_strength), bias_regularizer=l2(l2_str))(x)
	x=tf.keras.layers.BatchNormalization()(x)
	x=tf.keras.layers.ReLU()(x)
	x= tf.keras.layers.MaxPooling2D((2, 2))(x) 
        
 	# Seventh conv block
	x=tf.keras.layers.Conv2D(1024, (2, 2),strides=1, padding='valid',use_bias=False,
              kernel_initializer=tf.keras.initializers.HeNormal(),
              kernel_regularizer=tf.keras.regularizers.l2(l2_strength), bias_regularizer=l2(l2_str))(x)
	x=tf.keras.layers.BatchNormalization()(x)
	x=tf.keras.layers.ReLU()(x)
	# ATTENTION LAYER
	y=tf.keras.layers.Conv2D(num_classes, (1, 1),kernel_regularizer=tf.keras.regularizers.l2(l2_strength), bias_regularizer=l2(l2_str),use_bias="False")(x) #activation='softmax'
	y = tf.keras.layers.BatchNormalization()(y)
	y = tf.keras.layers.Softmax()(y)
	y = ScaleLayer()(y)
	z = tf.keras.layers.Conv2D(num_classes, (1, 1),kernel_regularizer=tf.keras.regularizers.l2(l2_strength), bias_regularizer=l2(l2_str),use_bias="False")(x) #,activation='sigmoid'
	z = tf.keras.layers.BatchNormalization()(z)
	z = tf.keras.layers.Activation(activation='sigmoid')(z)
	w = tf.keras.layers.multiply([y,z])
	w = SumLayer()(w)
	w = tf.keras.layers.Flatten()(w)
	#w = tf.keras.layers.Dense(1024,activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(),kernel_regularizer=tf.keras.regularizers.l2(l2_strength),bias_regularizer=l2(l2_str))(w) ##added to increase complexity
	#w = tf.keras.layers.Dropout(0.5)(w)
	w = tf.keras.layers.Dense(512,activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(),kernel_regularizer=tf.keras.regularizers.l2(l2_strength),bias_regularizer=l2(l2_str))(w) ##added to increase complexity
	w = tf.keras.layers.Dense(num_classes,activation='sigmoid')(w)
	model = Model(inputs=inputs, outputs=w)
	print('MODEL HAS {} PARAMETERS.'.format(model.count_params()))
	return model




def model_tf2(num_classes,l2_str=0):
	model = tf.keras.Sequential([
	tf.keras.layers.Conv2D(16, (3, 3),input_shape=(None, None, 1),strides=1, padding='same'),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.ReLU(),

	tf.keras.layers.Conv2D(16, (3, 3),strides=1, padding='same'),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.ReLU(),

	tf.keras.layers.MaxPooling2D((2, 2)),

# Second conv block
	tf.keras.layers.Conv2D(32, (3, 3),strides=1, padding='same'),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.ReLU(),

	tf.keras.layers.Conv2D(32, (3, 3),strides=1, padding='same'),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.ReLU(),

	tf.keras.layers.MaxPooling2D((2, 2)), 

# Third conv block        
	tf.keras.layers.Conv2D(64, (3, 3),strides=1, padding='same'),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.ReLU(),

	tf.keras.layers.Conv2D(64, (3, 3),strides=1, padding='same'),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.ReLU(),

	tf.keras.layers.MaxPooling2D((2, 2)), 
        
        
  
# Fourth conv block
	tf.keras.layers.Conv2D(128, (3, 3),strides=1, padding='same'),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.ReLU(),

	tf.keras.layers.Conv2D(128, (3, 3),strides=1, padding='same'),
	tf.keras.layers.ReLU(),
	tf.keras.layers.BatchNormalization(),

	tf.keras.layers.MaxPooling2D((2, 2)), 
        
# Fifth conv block
	tf.keras.layers.Conv2D(512, (3, 3),strides=1, padding='same'),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.ReLU(),

	tf.keras.layers.Conv2D(512, (3, 3),strides=1, padding='same'),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.ReLU(),

	tf.keras.layers.MaxPooling2D((2, 2)), 
        
        
# Sixth conv block
	tf.keras.layers.Conv2D(512, (3, 3),strides=1, padding='same'),
	tf.keras.layers.BatchNormalization(),  
	tf.keras.layers.ReLU(),

	tf.keras.layers.Conv2D(512, (3, 3),strides=1, padding='same'),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.ReLU(),

	tf.keras.layers.MaxPooling2D((2, 2)), 
        
 # Seventh conv block
	tf.keras.layers.Conv2D(1024, (2, 2),strides=1, padding='valid'),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.ReLU(),          
 # Eighth conv block
	tf.keras.layers.Conv2D(num_classes, (1, 1),activation='sigmoid'),
 #Final BLock
	tf.keras.layers.GlobalAveragePooling2D()]) 
    ## a fully covolutional NN !
	print('MODEL HAS {} PARAMETERS.'.format(model.count_params()))
	return model
	
def cutout(x, n_holes=1, length=40):   #time-freq masking
	"Cut out `n_holes` number of rectangular bands of size `length` in image at random locations."
	h,w = x.shape
	#print(h,w)
	for n in range(n_holes):
		h_y = np.random.randint(0, h)
		h_x = np.random.randint(0, w)
        	#print(h_x,h_y)
		y1 = int(np.clip(h_y - length / 2,0, h))
		y2 = int(np.clip(h_y + length / 2,0, h))
		x1 = int(np.clip(h_x - length / 2,0, w))
		x2 = int(np.clip(h_x + length / 2,0, w))
		x[:, y1:y2] = 0
		x[x1:x2,:] = 0
	return x
	
def mixup_specs(x1,y1,x2,y2):
	alpha = np.random.beta(1,1,1)
    	#print('alpha',alpha)
	x_new = alpha*x1 + (1-alpha)*x2
	y_new = alpha*y1 + (1-alpha)*y2
	##y_new = y1 + y2  # experimenting with not scaling down the labels
	return x_new,y_new

def mixup_func(mixup,X_train,y_train):
	len_mixup = len(mixup)
	if(len_mixup >1):
		for w in range(len_mixup-1):
			if(len_mixup ==1):
				break
			mixup_files = mixup[w],mixup[w+1]
			idx1 = mixup_files[0][1]
			idx2 = mixup_files[1][1]
			data_idx1 = X_train[idx1]
			data_idx2 = X_train[idx2]
			label_idx1 = y_train[idx1]
			label_idx2 = y_train[idx2]
			new_data,new_label = mixup_specs(data_idx1,label_idx1,data_idx2,label_idx2)
			new_data -= new_data.min()
			new_data /= new_data.max()
			X_train[idx1] = new_data
			y_train[idx1] = new_label
	return X_train,y_train

def train_generator(samples,labels,le,batch_size=32,n_classes=397):
	num_samples = len(samples)
	idxs = np.arange(num_samples)
	while True: # Loop forever so the generator never terminate
		shuffle(idxs)
		for offset in range(0, num_samples, batch_size):
                                # Get the samples you'll use in this batch
			idx = idxs[offset:offset+batch_size]
			batch_samples = [samples[x] for x in idx]
			batch_labels = np.array([labels[x] for x in idx])
                        # Initialise X_train and y_train arrays for this batch
			X_train =  [] 
			y_train = [] 
			mixup=[]
            		# For each example
			for p,ID in enumerate (batch_samples):
				try:
					audio_path,secondary_labels = ID
					spec = Image.open(audio_path) # here ID is path
                        		# Convert to numpy array
					spec = np.array(spec, dtype='float32')
					if(spec.shape != (128,625)):  # checking the size of the spectrogram
						continue
                        		# Normalize between 0.0 and 1.0
                        		# and exclude samples with nan
					spec -= spec.min()
					spec /= spec.max()
					if not spec.max() == 1.0 or not spec.min() == 0.0:
						continue
					#power_range =np.arange(1.0,3.0,0.5)
					#spec_pow = np.random.choice(power_range)
					##spec_pow_noisy =  have twp spec powers for noisy and for rest, appply on all at random 
					#if(("time" in ID) or ("freq" in ID)):
						#toss = np.random.binomial(1,0.5)
						#if(toss):
							#spec = spec**spec_pow   # no further increasin the noise
					##if(not("pink" in ID) or ("gauss" in ID) or ("time" in ID) or ("freq" in ID)):  # only supposedly pure samples
						##toss_cutout = np.random.binomial(1,0.5)
						##if(len(secondary_labels) == 0):  ##spec masking for sec labels less 1 ie for single_labels
							##if(toss_cutout):
								##spec = cutout(spec,2)
					spec = np.expand_dims(spec, -1)
					spec = np.expand_dims(spec, 0)
					##spec = np.repeat(spec, 3, axis=2)
                    			#Add new dimension for batch size
					if(len(X_train)==0):
						X_train = spec
						index = idx[p]
						y_tr = tf.keras.utils.to_categorical(labels[index], num_classes=n_classes)*1.0
						y_tr = np.expand_dims(y_tr,0)
						if(len(secondary_labels)> 0):
							sec_label_indices = le.transform(secondary_labels)
							y_tr[:,sec_label_indices] = 1.0 ## 1.0 worked well
						y_train = y_tr
					else:
						X_train = np.append(X_train,spec,axis=0)
						index = idx[p]
						y_tr = tf.keras.utils.to_categorical(labels[index], num_classes=n_classes)*1.0
						y_tr = np.expand_dims(y_tr,0)
						if(len(secondary_labels) > 0):
							sec_label_indices = le.transform(secondary_labels)
							y_tr[:,sec_label_indices] = 1.0  ##increased from 0.25 to 1.0 worked well ,lets try 0.50
						y_train = np.append(y_train,y_tr,axis=0)
					##if(not(('gauss' in ID) or ('pink' in ID))):
						##if(len(secondary_labels) ==0):
							##toss2 = np.random.binomial(1,0.40)  # since we are working wiht secondary labels , reducing chance of mixup
							##if(toss2):
								##mixup.append((ID,len(X_train)-1)) 
				except Exception as e:
                    			#print(e)
					continue
			##mixup_func(mixup,X_train,y_train) ##function to perform mixups based on beta distribution
			yield X_train,y_train
#make sure we get uniqe secondary labels and no repetion with primary label

if __name__=='__main__': 
	print("IN MAIN")
	df = pd.read_csv("./train_metadata.csv")
	df["secondary_labels"] = df["secondary_labels"].apply(literal_eval)
	df2 = df[['filename','primary_label','rating','secondary_labels']]
	print(df2.head())
	df2.set_index('filename',inplace=True)
	print(df2.head())
	dict_df  = df2.T.to_dict('list') 
	dicts = os.listdir("./FOR_EXP2/")
	base_path = "./FOR_EXP2/"
	specs = dicts
	files_path = []
	max_files= 2500
	repeated = 0
	for d in tqdm(dicts):
		files = os.listdir(base_path+d)
		files_partial = random.sample(files,int(1.0*len(files)))
		j = 0
		individual_paths = []
		for file in files_partial:
			spec_not_included = 0
			if("freq" in file):
				continue
			if(j > max_files):
				break
			main_file = file.split("_")[0] +".ogg"
			#check for ratings also
			primary_label,rating,secondary_labels = dict_df[main_file]
			if(rating <4.0):
				continue
			if(len(secondary_labels) >0):
				#print("before",secondary_labels)
				secondary_labels = list(set(secondary_labels)) ## removing duplicate entries
				to_remove = []
				for x in secondary_labels:
					if(x not in specs):
						to_remove.append(x)
						spec_not_included =1  # removing any spect with species not present in the specs(397 birds)
						#break				
				if(len(to_remove) > 0):					
					for x in to_remove:
						secondary_labels.remove(x)
				if(len(secondary_labels) >0):
					if(primary_label in secondary_labels):
						secondary_labels.remove(primary_label)
                    				##removinf if primary label is present in sec_labels
				#print("after",secondary_labels)
			if(not spec_not_included):
				complete_path = base_path+d+"/"+file
				individual_paths.append((complete_path,secondary_labels))
				j +=1
		idxs = np.arange(len(individual_paths))
		if(len(individual_paths) < max_files):                                       ##simple oversampling strategy
			repeated +=1
			ind_path_oversample_idxs = np.random.choice(idxs,max_files)
			ind_path_oversample = np.array(individual_paths)[ind_path_oversample_idxs].tolist()
		else:
			ind_path_oversample = individual_paths
		files_path.extend(ind_path_oversample)
	print("NUM_REPEATED",repeated)
	print(len(files_path))
	print(files_path[:10])
	y= [x[0].split("/")[2] for x in files_path]

	print(len(y),len(files_path),y[:10])
	le = LabelEncoder()
	le.fit(dicts)
	labels = le.transform(y)
	print(labels[:10])
	files_train,files_val,labels_train,labels_val = train_test_split(files_path,labels,test_size=0.05,stratify=labels,random_state=1920)
	print(len(files_train),len(files_val))
	#model = model_tf_att(num_classes=len(dicts)) #add l2 regularization
	#model = model_tf2(num_classes=len(dicts)) #add l2 regularization
	#model = model_tf_multiatt(len(dicts))
	model = model_mult_att_CNN(len(dicts))
	print(model.summary())
	num_classes = len(dicts)


	# Compile the model and specify optimizer, loss and metric
	model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss=BinaryFocalLoss(gamma=2,label_smoothing=0.025), #				#tfa.losses.SigmoidFocalCrossEntropy(alpha= 0.25,gamma= 2.0), #tf.keras.losses.BinaryCrossentropy(label_smoothing=0.025)
              metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),
								tf.keras.metrics.AUC(curve='PR',multi_label=False),
								tfa.metrics.F1Score(average='micro',num_classes=num_classes,threshold=0.50)])
			#tfa.metrics.F1Score(num_classes=len(dicts),average='micro',threshold= 0.25,name= 'f1_score_thresh_0.25'),
			#tfa.metrics.F1Score(num_classes=len(dicts),average='micro',threshold=0.50,name = 'f1_score_thresh_0.50') ]) ##remove accuracy from the metrics
	                 # try to use multiple instances with different thresholds!
			#Add callbacks to reduce the learning rate if needed, early stopping, and checkpoint saving
	callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                  patience=2, 
                                                  verbose=1, 
                                                  factor=0.5),
	tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                              verbose=1,
                                              patience=5),
	tf.keras.callbacks.ModelCheckpoint(filepath='./logs_all_with_augs/exp9/2500_files_with_multi_02_cnn_att_sec_labels_1.0_label_smthng_with_focal_loss_rating_4.0.h5', 
                                                monitor='val_loss',
                                                verbose=0,
                                                save_best_only=True),
	tf.keras.callbacks.TensorBoard(log_dir= "./logs_all_with_augs/exp9",
                                        histogram_freq=1,
                                        write_graph = True,
                                        write_images=False,
                                        profile_batch=0)]  # for tensorflow >= 2.0


	num_batch_size=32
	num_classes = len(dicts)
	print("NUM CLASSES",num_classes)
	train_gen = train_generator(files_train,np.array(labels_train),le,batch_size=32,n_classes=num_classes)

	val_gen = train_generator(files_val,np.array(labels_val),le,batch_size=32,n_classes=num_classes) ##  data augmentation here also if done in train set

	model.fit(train_gen,
          validation_data=val_gen,
          epochs=50,
          verbose=1,
          steps_per_epoch= int(len(files_train)//num_batch_size),
          validation_steps=int(len(files_val)//num_batch_size),
          callbacks=callbacks)




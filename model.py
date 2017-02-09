import os
import numpy as np
import cv2
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda, SpatialDropout2D, Dropout, Convolution2D, Cropping2D, MaxPooling2D
from keras.optimizers import Adam
import csv
from sklearn.model_selection import train_test_split
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import random

DEBUG = False
DROPOUT = False
LAMBDA_USAGE = True

batch_size = 256
resizing_measures = (200,66)

nb_epoch = 5
dropout_prob = .25
steer_offset = 0.25  #this define the value to be added to the steering angle for flippinf purposes

labels_path = "E:/Self Driving Car Nanodegree/Behavioural cloning/data"
labels_file_name = "driving_log.csv"
processed_labels_file_name = "new_driving_log_zeroReduction.csv"
new_labels_file_name = "new_driving_log_test.csv"
images_path = "E:/Self Driving Car Nanodegree/Behavioural cloning/data/IMG"
flipped_images_path = "E:/Self Driving Car Nanodegree/Behavioural cloning/data/Flipped"
debug_labels_path = "E:\Self Driving Car Nanodegree\Behavioural cloning\Tmp Data"

##########################################################################################################################################################################
################################################					  		DEBUG PURPOSES only 	 						#################################################
###########################################################################################################################################################################


### Used only to verify the recomputed steering csv file.
def verify_sets(X_train,X_validation,X_test,y_train,y_validation,y_test):

	error = False
	images = ['IMG/center_2016_12_01_13_30_48_404.jpg', 'IMG/left_2016_12_01_13_30_48_404.jpg', 'IMG/right_2016_12_01_13_30_48_404.jpg', 'IMG/center_2016_12_01_13_39_13_263.jpg', 
	'IMG/left_2016_12_01_13_39_13_263.jpg', 'IMG/right_2016_12_01_13_39_13_263.jpg', 'IMG/center_2016_12_01_13_39_27_924.jpg', 'IMG/left_2016_12_01_13_39_27_924.jpg', 
	'IMG/right_2016_12_01_13_39_27_924.jpg', 'IMG/center_2016_12_01_13_32_57_203.jpg', 'IMG/left_2016_12_01_13_32_57_203.jpg', 'IMG/right_2016_12_01_13_32_57_203.jpg', 
	'IMG/center_2016_12_01_13_32_57_304.jpg', 'IMG/left_2016_12_01_13_32_57_304.jpg', 'IMG/right_2016_12_01_13_32_57_304.jpg','IMG/center_2016_12_01_13_32_47_394.jpg','IMG/center_2016_12_01_13_32_48_200.jpg','IMG/center_2016_12_01_13_32_52_551.jpg','IMG/center_2016_12_01_13_32_53_156.jpg',
	'IMG/center_2016_12_01_13_32_53_459.jpg','IMG/center_2016_12_01_13_32_59_427.jpg','IMG/center_2016_12_01_13_33_05_599.jpg','IMG/center_2016_12_01_13_33_14_433.jpg',
	'IMG/center_2016_12_01_13_33_15_544.jpg','IMG/center_2016_12_01_13_33_45_117.jpg','IMG/center_2016_12_01_13_33_54_272.jpg','IMG/center_2016_12_01_13_34_07_970.jpg',
	'IMG/center_2016_12_01_13_34_11_104.jpg','IMG/center_2016_12_01_13_40_09_658.jpg','IMG/center_2016_12_01_13_40_56_740.jpg','IMG/center_2016_12_01_13_41_02_629.jpg']
	steering_val = [0,0+steer_offset,0-steer_offset,0,0+steer_offset,0-steer_offset,-0.9332381,-0.9332381+steer_offset,-0.9332381-steer_offset,
					0.1670138,0.1670138+steer_offset,0.1670138-steer_offset,0.3488158,0.3488158+steer_offset,0.3488158-steer_offset,0.3583844,-0.05975719, -0.2306556, 
					0.1574452,0.2531306, 0.100034, 0.1765823,0.3488158,-0.1547008,0.4540697,-0.2306556,0.08089697,-0.08824026,0.5880292, -0.0787459,-0.4395315]
	for index in range(len(images)):
		try:
			assert(y_train[X_train.index(images[index])] == steering_val[index])
		except AssertionError as e:
			print("Assertion Error on image name {}".format(images[index]))
		except Exception as e:
			print(e)
			error = True
			pass

		try:
			assert(y_validation[X_validation.index(images[index])] == steering_val[index])
		except AssertionError as e:
			print("Assertion Error on image name {}".format(images[index]))
		except Exception as e:
			print(e)
			error = True
			pass

		try:
			assert(y_test[X_test.index(images[index])] == steering_val[index])
		except AssertionError as e:
			print("Assertion Error on image name {}".format(images[index]))
		except Exception as e:
			print(e)
			error = True
			pass
	if error:
		print("Sets are not created correctly")
	else:
		print("Sets verified correctly")

def DEBUG_load_dataset():

	X_train = np.array([])
	y_train = np.array([0,0.5784606,-0.2306556,0,0.2876218,-0.243562])

	X_test = np.array([])
	y_test = np.array([0,0.5784606,-0.2306556,0,0.2876218,-0.243562])

	X_validation = np.array([])
	y_validation = np.array([0,0.5784606,-0.2306556,0,0.2876218,-0.243562])

	files = os.listdir(debug_labels_path)
	
	for f in files:
		image = Image.open("{}/{}".format(debug_labels_path,f))
		#image = cv2.imread("{}/{}".format(debug_labels_path,f))
		"""
		if not LAMBDA_USAGE:
			dim = (200,66)
			image = cv2.resize(image, resizing_measures, interpolation = cv2.INTER_AREA)
			image = normalization_TMP(image)
		"""
		image = np.asarray(image).astype(float)

		if(X_train.size == 0):
			X_train = np.array([image])
			X_test = np.array([image])
			X_validation = np.array([image])
		else:
			X_train = np.insert(X_train,X_train.shape[0],image,axis = 0)
			X_test = np.insert(X_test,X_test.shape[0],image,axis = 0)
			X_validation = np.insert(X_test,X_test.shape[0],image,axis = 0)
	return X_train,X_validation,X_test,y_train,y_validation,y_test

def DEBUG_load_dataset_names():

	X_train = []
	y_train = np.array([0,0.5784606,-0.2306556,0,0.2876218,-0.243562])

	X_test = []
	y_test = np.array([0,0.5784606,-0.2306556,0,0.2876218,-0.243562])

	X_validation = []
	y_validation = np.array([0,0.5784606,-0.2306556,0,0.2876218,-0.243562])

	files = os.listdir(debug_labels_path)
	
	for f in files:

		X_train.append(f)
		X_test.append(f)
		X_validation.append(f)

	return np.asarray(X_train),np.asarray(X_validation),np.asarray(X_test),y_train,y_validation,y_test


###########################################################################################################################################################################
###########################################################################################################################################################################

#Used to flip the images. Used once offline

def flip_images():
	print("Flipping images process started")
	files = os.listdir(images_path)
	i=0
	for f in files:
		if(i%100 == 0):
			print("Processed {} images".format(i))
		i+=1
		image = mpimg.imread(images_path+"/"+f)

		flipped_image = cv2.flip(image, 1)
		flipped_image = cv2.cvtColor(flipped_image,cv2.COLOR_BGR2RGB)
		cv2.imwrite(flipped_images_path+"/"+f.split('.')[0]+"_flipped.jpg",flipped_image)
		cv2.destroyAllWindows()
	print("Flipping images process finished")


def preprocess_labels():
	print("Preprocessing labels file started")
	
	with open(labels_path+"/"+labels_file_name, 'rt') as f:
		reader = csv.reader(f)
		raw_labels_list = list(reader)

	images_names = []
	steering_vals = []
	for i in range(1,len(raw_labels_list)):
		
		steering_val = float(raw_labels_list[i][3])

		center_image_name = raw_labels_list[i][0].strip()
		left_image_name = raw_labels_list[i][1].strip()
		right_image_name = raw_labels_list[i][2].strip()
		
		### First Reduce the number of examples with 0 steering angle
		if(abs(steering_val) == 0.0):
			if(np.random.binomial(1, .04, 1)):
				images_names.append(center_image_name)
				images_names.append(left_image_name)
				images_names.append(right_image_name)
				steering_vals.append(steering_val)
				steering_vals.append(steering_val + steer_offset)
				steering_vals.append(steering_val - steer_offset)

				images_names.append(center_image_name.split('.')[0]+"_flipped.jpg") 
				images_names.append(left_image_name.split('.')[0]+"_flipped.jpg") 
				images_names.append(right_image_name.split('.')[0]+"_flipped.jpg") 
				steering_vals.append(-steering_val)
				steering_vals.append(-steering_val + steer_offset)
				steering_vals.append(-steering_val - steer_offset)
				
		else:
			images_names.append(center_image_name)
			images_names.append(left_image_name)
			images_names.append(right_image_name)
			steering_vals.append(steering_val)
			steering_vals.append(steering_val + steer_offset)
			steering_vals.append(steering_val - steer_offset)

			images_names.append(center_image_name.split('.')[0]+"_flipped.jpg") 
			images_names.append(left_image_name.split('.')[0]+"_flipped.jpg") 
			images_names.append(right_image_name.split('.')[0]+"_flipped.jpg") 
			steering_vals.append(-steering_val)
			steering_vals.append(-steering_val + steer_offset)
			steering_vals.append(-steering_val - steer_offset)

	return images_names,steering_vals



def normalize_image(image):
	return image / 255.0 - 0.5
	

def resize(image):
	import tensorflow as tf
	return tf.image.resize_images(image, (66, 200))

def create_model():

	model = Sequential()

	if LAMBDA_USAGE:
		### Images preprocessing: resize and normalization.
		model.add(Cropping2D(cropping=((60, 0), (0, 0)), input_shape=(160,320,3)))

		model.add(Lambda(resize))

		#model.add(Lambda(normalize_image,input_shape=(66,200,3)))
		model.add(Lambda(normalize_image))

		### 1st conv. layer: 24 5x5 Filters with 2x2 stride and padding "valid" 
		model.add(Convolution2D(24, 5, 5, activation = "elu", border_mode='valid', init="he_normal"))
	else:
		### 1st conv. layer: 24 5x5 Filters with 2x2 stride and padding "valid" 
		model.add(Convolution2D(24, 5, 5, activation = "elu", border_mode='valid', init="he_normal",input_shape=(66,200,3)))
	
	### SpatialDropout 2D after the convolutional layer
	if DROPOUT:
		model.add(Dropout(dropout_prob))
	model.add(MaxPooling2D(pool_size=(2,2),border_mode='same'))

	### 2nd conv. layer: 36 5x5 Filters with 2x2 stride and padding "valid" 
	model.add(Convolution2D(36, 5, 5, activation = "elu", border_mode='valid', init="he_normal"))
	### SpatialDropout 2D after the convolutional layer
	if DROPOUT:
		model.add(Dropout(dropout_prob))
	model.add(MaxPooling2D(pool_size=(2,2),border_mode='same'))


	### 3rd conv. layer: 48 5x5 Filters with 2x2 stride and padding "valid" 
	model.add(Convolution2D(48, 5, 5, activation = "elu", border_mode='valid', init="he_normal"))
	### SpatialDropout 2D after the convolutional layer
	if DROPOUT:
		model.add(Dropout(dropout_prob))
	model.add(MaxPooling2D(pool_size=(2,2),border_mode='same'))


	### 4th conv. layer: 64 3x3 Filters with no stride and padding "valid" 
	model.add(Convolution2D(64, 3, 3, activation = "elu", border_mode='valid', init="he_normal"))
	### SpatialDropout 2D after the convolutional layer
	if DROPOUT:
		model.add(Dropout(dropout_prob))


	### 5th conv. layer: 64 3x3 Filters with no stride and padding "valid" 
	model.add(Convolution2D(64, 3, 3, activation = "elu", border_mode='valid', init="he_normal"))
	### SpatialDropout 2D after the convolutional layer
	if DROPOUT:
		model.add(Dropout(dropout_prob))
	

	model.add(Flatten())

	### 6th layer: fully layer. 100 output value
	model.add(Dense(100, activation = "elu", init="he_normal"))
	### SpatialDropout 2D after the convolutional layer
	if DROPOUT:
		model.add(Dropout(dropout_prob))

	### 7th layer: fully layer. 50 output value
	model.add(Dense(50, activation = "elu", init="he_normal"))
	### SpatialDropout 2D after the convolutional layer
	if DROPOUT:
		model.add(Dropout(dropout_prob))

	### 8th layer: fully layer. 10 output value
	model.add(Dense(10, activation = "elu", init="he_normal"))
	### SpatialDropout 2D after the convolutional layer
	if DROPOUT:
		model.add(Dropout(dropout_prob))

	### output: 1 value
	model.add(Dense(1,init="he_normal",activation="tanh"))

	#model.compile('adam', 'categorical_crossentropy', ['accuracy'])
	model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=0.001),
              metrics=['mean_squared_error'])
	return model	

def normalization_TMP(image):
	new_img = (image-128.0)/128.0
	return new_img



def generator(image_names, labels, batch_size):
	
	start = 0
	end = start + batch_size
	n = len(image_names)
	while True:
		X_batch = np.array([])
		for image_name in image_names[start:end]:
			path = images_path[:-4]
			if DEBUG:
				path = debug_labels_path
			
			image = Image.open("{}/{}".format(path,image_name))
			image = np.asarray(image)
			
			if(X_batch.size == 0):
				X_batch = np.array([image])
			else:
				X_batch = np.insert(X_batch,X_batch.shape[0],[image],axis = 0)
		y_batch = np.array(labels[start:end])
		start += batch_size
		end += batch_size
		if start >= n:
			start = 0
			end = batch_size
		yield (X_batch, y_batch)
    

def split_dataset(images_names, steering_vals):

	###Steering_classes is used to balance the train/test/validation from steering angle poing of view

	steering_classes = []
	for steering_val in steering_vals:
		steering_classes.append(0 if steering_val == 0 else -1 if steering_val <0 else 1)

	X_train, X_validation_test, y_train, y_validation_test = train_test_split(images_names, steering_vals, test_size=0.3, stratify=steering_classes)
	steering_classes = []

	for steering_val in y_validation_test:
		steering_classes.append(0 if steering_val == 0 else -1 if steering_val <0 else 1)

	X_validation, X_test, y_validation, y_test = train_test_split(X_validation_test, y_validation_test, test_size=0.33, stratify=steering_classes)

	return np.asarray(X_train),np.asarray(X_validation),np.asarray(X_test),np.asarray(y_train),np.asarray(y_validation),np.asarray(y_test)


def run_model(X_train,X_test,y_train,y_test,X_validation = np.array([]),y_validation= np.array([])):

	model = create_model()

	if DEBUG:

		training_history = model.fit_generator(
			generator(X_train, y_train, batch_size),
			samples_per_epoch = len(X_train),
			nb_epoch=nb_epoch, 
			validation_data = generator(X_validation, y_validation, batch_size),
			nb_val_samples = len(X_validation),verbose = 1
	    )

	if not DEBUG:
		
		training_history = model.fit_generator(
			generator(X_train, y_train, batch_size),
			samples_per_epoch = len(X_train),
			nb_epoch=nb_epoch, 
			validation_data = generator(X_validation, y_validation, batch_size),
			nb_val_samples = len(X_validation),verbose = 1
	    )
		
		print("Testing the model with the testset...")

		test_score = model.evaluate_generator(
			generator(X_test, y_test,batch_size), 
			val_samples = len(X_test), 
			max_q_size=10, 
			nb_worker=1, 
			pickle_safe=False)
		print(test_score)
	
	print("Saving model and weights...")

	test_string = "model_DBG-{}_DRPT-{}_DropProb-{}_E-{}_steeroffset-{}_BtchSize-{}_Lambda-{}".format(DEBUG,DROPOUT,dropout_prob,nb_epoch,steer_offset,batch_size,LAMBDA_USAGE)
	model.save("{}.h5".format(test_string))  # creates a HDF5 file 'my_model.h5'
	model_json = model.to_json()
	with open("{}.json".format(test_string), "w") as json_file:
		json_file.write(model_json)
		
if __name__ == '__main__':

		#### Used Only once offline to flip the entire dataset.
		#flip_images() 
		####

		images_names, steering_vals = preprocess_labels()

		if DEBUG:
			X_train,X_validation,X_test,y_train,y_validation,y_test = DEBUG_load_dataset_names()
		else:	
			X_train,X_validation,X_test,y_train,y_validation,y_test = split_dataset(images_names, steering_vals)
		
		run_model(X_train,X_test,y_train,y_test,X_validation,y_validation)
	

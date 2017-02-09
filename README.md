# Self Driving Car NanoDegree - Udacity
# Project 3 - Behavioural cloning

## Overview

The goal for this third project for the nanodegree is to model a Deep Neural Network able to clone the human behaviour in driving along a track. In this particular scenario, the behaviour to clone comes from a simulator.
The dataset used for this project does not come from the simulation in the simulator, but the provided dataset was used. 
The model was first trained with a part of this dataset (training set), and then tested in the "autonomous" function of the simulator.

## Preliminare data analisys

The dataset provided by udacity is made of 8036 per camera coming from the first of the two tracks of the simulator. This number of examples was considered sufficient as starting number for the purposes of this project.

Track #1, from where the images comes from, has a prevalency of left corner. This bring the dataset to appears unbalanced from positive and necative steering angle (right and left steering angles rispectively). 

Moreover, the dataset appear unbalanced also from distribution of the steering angles: 

1- The number of examples with 0 steering angle are is huge respect to the != 0 angles. 
2- The number of examples of steering angle decrease with the increase of the steering angle itself (both for positive and negative angles)

This behaviour was expected since the majority of the track is straight and because the normal behaviour is to drive straight avoiding strong turn (using small steering values for more time)

# Image Preprocessing and data Augmentation

For the purposes of the project I decided to start from the dataset provided by Udacity. I considered the 8036 pictures for each camera sufficient for the training/validation/test. 
First step to complete this project was *data analysis* and *data augmentation*. In order to balance the values of steering in the dataset, I flipped all the images along y axis (from all the cameras) recomputing the steering angle accordingly (*-1). 

Therefore steering angles for images coming from the lateral cameras were computed in order to teach the model how to "recover" the car position away from the center of the road. To accomplish this goal perfectly, information about geometry of the scene would have been required. Since nor car quotes neither distance between the cameras was known, some tests with different valus of steering angle were done.
First a fixed value of steering angle offset was used and added to the images. 
After a verification about the distribution of the steering angle in the dataset, I figure out that the dataset was not balanced. Examples were mostly with steering angle 0 and using a fixed offset for left and right cameras bringbore to 3 very high peaks in the distribution. So I decided to first select just parts of the images with steering angle = 0 and to add an offest to left and right cameras sampled from a Normal distribution with mean = 0.25 and variance 0.1.

This bring me to a steering angle distribution like the following:

 ![Alt text](E:\Self Driving Car Nanodegree\Behavioural cloning\figure_1.png?raw=true "Optional Title")

Thats not perfect. The ideal condition is a uniform distribution around the 0.

# Model Definition and image processing

The model was inspired from the Nvidia CNN suggested. A series of 5 convolutional layers followed by three fully connected layer. First three convolutional layers were desined to apply 5x5 filters with strid of size (2,2), while the last two decrese the filter size to 3x3 with stride (1x1).

Before to provide the training samples to the network, images were resized and normalized.
*reshape* of images was used to reduce the number of parameters to the same number of Nvidia CNN. 
*normalization* was applied to the images in order to simplify the complexity of the network and help the optimization algorithm convergency.

In order to prevent from overfitting, also dropout layers were added after all the layers (convolutional and fully connected). The dropout probability was fixed to 0.25 after some tests. 

# Model Training 

Finally, before the training procedure, the dataset was split in 3 sets: Test set = 10% of the total dataset; training set = 80% of the total dataset prived of the test set; validation set = 20% of the total dataset prived of the test set

The model was first trained with the support of the generator (in order to prevent memory issue due to the load of the entire training/validation set), and than evaluated over the test set always using the generator defined. Adam optimizer was used together with the mean_squared_Error loss function. For adam optimizer, a value of .001 learning rate was used as starting learning rate value.
Batch size was fixed to 256 (value that fit the memory of my laptop without decreasing of performances).
5 Epochs were used for optimization puposes, even if at the 3 one the algorithm already converges.

# Parallel tests	

In parallel to all the verification described above, I tryed to learn the CNN also with grayscale images and cropped images. However, the difference in performance and time request with both these two "size reduction" were not so high. And since performances with full RGB images were better, I decided to keep images as they were.

# Interesting Optimization for future

Unfortunately timings requirements does not let me to test some different solutions. However I would like to place the attention (and asking for some suggestions) about what could be very useful for these systems.

### 1- Data Augmentation

### 2- Googlenet approach with inception module

### 3- RNN

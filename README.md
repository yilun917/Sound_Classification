# Sound Classification

## Package Needed
Keras
TensorFlow
pickle
librosa
struct
sklearn
datetime
matplotlib
os

## Project Overview
The objective of the project is to build and train a model to identify sound. 

## File Structure
1. [Audio Classification.ipynb](./Audio&#32;Classification.ipynb) is the main file with all the functions and model definitions and most of the results.  
2. [sound_classification_model.json](./sound_classification_model.json) is the Model Object where the Neural Network is defined and store to .h5 file to avoid repeated model   compilation.  
3. [weights.best.basic_cnn.hdf5](./weights.best.basic_cnn.hdf5) is the file that contains the weight used in the model. So that we can quickly continue the progress to do a prediction.
4. [UrbanSound8K_README.txt](./UrbanSound8K_README.txt) is the README of the original dataset.  
5. [audio_features_mfcc40.pickle](./audio_features_mfcc40.pickle) is the preprocessed data file. Processing the data can take 15 minutes up, store the preprocessed data is important to save time.  

## How to Use
At the end of the notebook, you have the part that you can load the model and the pretrained weights. And use the next cell to load user specified sound clip. The sound clip have to have about 2s, and the part containing information need to be within the 2s part. 

## Model Architecture
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 173x40x1 Audio Spectrum   							| 
| Convolution 2x2     	| 1x1 stride, valid padding, filter size 16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding 	|
| Dropout		 		|Droop rate 0.4									|
| Convolution 2x2     	| 1x1 stride, valid padding, filter size 32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding 	|
| Dropout		 		|Droop rate 0.4									|
| Convolution 2x2     	| 1x1 stride, valid padding, filter size 64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding  	|
| Dropout		 		|Droop rate 0.4									|
| Convolution 2x2     	| 1x1 stride, valid padding, filter size 128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding 	|
| Dropout		 		|Droop rate 0.4									|
| Flatten				| faltten to 1 dimension			|
| Softmax				|Output layer using Softmax				|
  
Over all the model has 4 Convolutional blocks. Each block is consisted of a convolution layer folowed by a ReLu activation layer. And then a Max pooling layer and a Dropout to regularize the ouput. Finally, there is a fully connected flatten layer with a softmax activation. The model is a bit like the classic LeNet-5 model.

## Results
![alt text][./Result.png]
The achieved training accuracy is 90% and the validation accuracy is 90%.

## Unsuccessful Attempts
MobileNet model is attempted, but not successful. The dimensions seems to be 

## Future Developments

## Liscense
Private use only

# Sound Classification

## Project Overview
The objective of the project is to build and train a model to identify sound. Due to the limitation of the obtained dataset, the Neural Network model is able to identify 10 types of sound including:  
0 = air_conditioner  
1 = car_horn  
2 = children_playing  
3 = dog_bark  
4 = drilling  
5 = engine_idling  
6 = gun_shot  
7 = jackhammer  
8 = siren  
9 = street_music  


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
  
Setting up a virtual environment and install the required packages in the virtual environment is recommended. So that, different projects will not affect each other.


## File Structure
1. [Audio Classification.ipynb](./Audio&#32;Classification.ipynb) is the main file with all the functions and model definitions and most of the results.  
2. [sound_classification_model.json](./sound_classification_model.json) is the Model Object where the Neural Network is defined and store to .h5 file to avoid repeated model   compilation.  
3. [weights.best.basic_cnn.hdf5](./weights.best.basic_cnn.hdf5) is the file that contains the weight used in the model. So that we can quickly continue the progress to do a prediction.
4. [UrbanSound8K](./UrbanSound8K) is the dataset obtained online to train the Neural Network model. There is also a [README](./UrbanSound8K/UrbanSound8K_README.txt) file in the folder which describes the dataset.  
5. [audio_features_mfcc40.pickle](./audio_features_mfcc40.pickle) is the preprocessed data file. Processing the data can take 15 minutes up, store the preprocessed data is important to save time.  

**In order for the program to run correctly, the UrbanSound8K folder should be unchanged and stayed in the main folder. The rest of the files need to be in the same folder as the notebook file. Otherwise, the path reference to some of the files in the notebook need to be adjusted accordingly.**


## Code Structure
All the code is contained in the jupyter notebook named "Audio Classification.ipynb".  
1. The first cell is used to import libraries.  
2. Cell 2 to 7 is used to extract the audio information and store them into local variables for the later part of the function to use. Also, the file is preprocessed and store in pickle file for easy access next time.  
3. Cell 8 to 12 is used to build Neural Network models in Keras. Note that there are 2 models, one LaNet-5 like CNN model and MobileNet model. Only the LaNet-5 like model works, the MobileNet model is experimented but did not work.  
4. Cell 13 to 18 is used to train and evaluate the model using the preproocessed data.  
5. Cell 19 is used to store the model and the trained weights.  
6. The cells after cell 19 is used to extract the previously built and trained model (another time when you open the jupyter notebook) and then use the model to make prediction on user uploaded data.  


## How to Use
The entire notebook contains the functions to preprocess the specified dataset, define a Convolutional Neural Network Model and to train the model with the data. The model and the trained weights are save into specified files.  
At the end of the notebook, you have the part that you can load the model and the pretrained weights. And use the next cell to load user specified sound clip. The sound clip have to have about 2s, and the part containing information need to be within the 2s part. 


## Data Preprocessing
The data preprocessing in this case is to turn the 1D audio signal to 2D (actually 3D, but the last dimension is 1) audio sprectrum. This is done through Librosa library.  
One critical parameter-the vertical dimention in the audio sprectrum (the dimention n in the input sprectrum n*173*1) is experimented.  
![alt text](./Model&#32;Accuracy&#32;vs.&#32;MFCC&#32;Number.png)  
As the image show, when the vertical dimention is 40 the model will have the highest accuracy. 


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
![alt text](./Result.png)
As the graph show, from training epoch of about 150 the accuracy improves very slowly, as the curve plateaus. More epoches than 150 may cause over fitting.Thus, the training epoch is set to 150. A slightly lower number (~120) is also acceptable.
The achieved training accuracy is ~93% and the validation accuracy is ~89% and the test accuracy is ~90%.

## Unsuccessful Attempts
MobileNet model is attempted to be applied, but not successful. The dimensions of the dataset seems to be inconsistent with the predifined model. The MobileNet model works best with square images (in this case audio sprectrum), but the data we have is 40x173x1 which is very streched. This could be the reason while MobileNet and most of other CNN models cannot be applied on the dataset. However, even the bug is fixed, it is likely that CNN models won't work well on audio data.

## Future Developments
As to the future development, one can either fix the issue with MobileNet (and other CNN networks) and have the model trained on the preprocessed data or use an Recurrent Neural Network (RNN) model. It is much more effective to RNN models on sequential data like audio or speech. Since RNN model can capture the sequential relation in the data, while CNN model does this poorly.

## Liscense
MIT Liscense

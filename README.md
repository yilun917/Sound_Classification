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
This is the [Main Notebook](./Audio&#32;Classification.ipynb) with all the functions and model definitions and most of the results.
This is the [Model Object](./model.h5) where the Neural Network is defined and store to .h5 file to avoid repeated model compilation.
This is the [README](./UrbanSound8K_README.txt) of the original dataset.
This is the [Preprocessed Data](./audio_features_mfcc40.pickle). Processing the data can take 15 minutes up, store the preprocessed data is important to save time.



## How to Use
At the end of the notebook, you have the part that you can load the model and the pretrained weights. And use the next cell to load user specified sound clip. The sound clip have to have about 2s, and the part containing information need to be within the 2s part. 

## Model Architecture


## Results

## Unsuccessful Attempts

## Future Developments

## Liscense
Private use only

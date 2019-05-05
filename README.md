This is a Convolutional Neural Network that has been trained using the Emnist_Balanced Dataset to recognise Handwritten Characters and has an accuracy of 89%.
To use the pre-trained model simply load the model present in the folder Model or use Main.py to use it as an api
It requires a normal Handwrtten Document with a plain background as input.
preprocessing.py will then perform preprocessing as well as segment the image for character recognition.
The following is required to run the code:
python 3.6
tensorflow
keras
opencv 4.0
Emnist_Balanced Dataset which can be downloaded from https://www.nist.gov/node/1298471/emnist-dataset or https://www.kaggle.com/crawford/emnist and must be added to the folder Train
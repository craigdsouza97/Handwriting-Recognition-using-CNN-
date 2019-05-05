# Description
This is a Convolutional Neural Network with 28 Convolutional Layers that has been trained using the Emnist_Balanced Dataset to recognise Handwritten Characters and has an accuracy of 89%.

# Usage
1. Using Pre-trained models
   - First extract the models from Models/Models.rar
   - The models require a 28X28 size binary image 
2. Using it as an api
   - First create an empty folder named 'Data'
   - Then extract the models from Models/Models.rar into Models
   - Run Main.py
   - It requires a normal Handwrtten Document with a plain background as input.
   - Images need to be encoded into base64 strings and sent via post 
   - preprocessing.py is used to perform preprocessing as well as segment the image for character recognition.

# Requirements
- python 3.6
- tensorflow
- keras
- flask (for using it as an api)
- opencv 4.0
- Emnist_Balanced Dataset which can be downloaded from https://www.nist.gov/node/1298471/emnist-dataset or https://www.kaggle.com/crawford/emnist and must be added to the folder Train

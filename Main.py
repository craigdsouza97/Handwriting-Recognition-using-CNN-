import cv2
import base64
from preprocessing import Preprocessing
from cnn import CNN,clearData
from flask import Flask, request
import jsonpickle
from keras.models import load_model
import os
# Initialize the Flask application
app = Flask(__name__)
model1 = load_model('./Models/hr1.h5')
print("Loaded 1st model")
model2 = load_model('./Models/hr2.h5')
print("Loaded 2nd model")
@app.route('/api/CHR', methods=['POST'])
def CHR():
    #convert string of image data to uint8
    a=jsonpickle.decode(request.data)
    img=a['image']
    b = bytes(img, 'utf-8')
    with open("Data/input.jpg", "wb") as fh:
        fh.write(base64.decodestring(b))
    img=cv2.imread('Data/input.jpg',0)
    lines=Preprocessing(img)
    if type(lines)!=str:
        print('Segmentation done')
        #output="Output:\n"+CNN(lines,model,1)
        output="OUTPUTS\n"
        output+="\nModel with 28 Convolutional Layers:\n"+CNN(lines,model1,1)
        output+="\nModel with 28 Convolutional Layers\nand Data Augmentation:\n"+CNN(lines,model2,1)
        clearData()
        print(output)
        try:
            os.mkdir('./Data/')
        except:
            print("Folder Exists")
        return ""+output
    else:
        return ""+lines

app.run(port=3000,threaded=False)


import cv2
import os
import pandas as pd
import shutil

def CNN(lines,model,n):
    print('Recognition')
    Characters=getCharacters(n)
    output=recognise(lines,model,Characters)
    return output

def prepare(path):
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    a,img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img=cv2.resize(img,(28,28))
    img = img.reshape([1,28,28,1])
    return img

def getCharacters(n):
    Characters=[]
    if n==1:
        Char=pd.read_csv('./Models/emnist-balanced-mapping.csv')
        for i in range(47):
            Characters.append(chr(Char['b'][i]))
    elif n==2:
        for i in range(ord('A'),ord('Z')+1):
            Characters.append(chr(i))
    print(Characters)
    return Characters


def recognise(lines,model,Characters):
    print(lines)
    s=""
    t=lines[0][0]
    for line in os.listdir('./Data/'):
        if(line!='input.jpg'):
            for words in os.listdir('./Data/'+str(line)):
                for image in os.listdir('./Data/'+str(line)+'/'+str(words)):
                    try:
                        classes = model.predict_classes([prepare('./Data/'+str(line)+'/'+str(words)+'/'+image)])
                        s+=""+Characters[int(classes)]
                    except:
                        s+=""
                s+=' '
            s+='\n'
    if len(s)==0:
        s="No text detected"
    return s

def clearData():
    try:
        shutil.rmtree('Data/')
    except:
        print('No Directory')
#%%

    
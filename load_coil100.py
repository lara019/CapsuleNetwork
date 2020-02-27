#https://www.kaggle.com/criscastromaya/cnn-for-image-classification-in-coil-100-dataset

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob,string
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img,img_to_array
import cv2
import matplotlib.pyplot as plt
import codecs
from tqdm import tqdm
import os,sys

import other_utils
import capsulenet_CIFAR10

import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from print import evaluate
from keras.datasets import cifar10

import pickle
import os

from sklearn.model_selection import train_test_split
from scipy import interpolate
from skimage.transform import resize
from print import visualize_example
import other_utils
from time import gmtime, strftime
from matplotlib import pyplot

from numpy import array
from numpy import argmax
from keras.utils import to_categorical


#path = 'datasets\\coil-100\\coil-100\\*.png'
numImages = 9
path = 'datasets/test/'
files1 = os.listdir(path)
cont = 0
for f in files1:	
    #print(os.path.splitext(f)[0])
    cont = cont + 1



#list files
#path = path + "*.png"
#files=glob.glob(path)
#print(files)
from matplotlib import pyplot

def resize28(arrayCompleto, w, h):
    arrayCompleto_nuevo = []
    arrayCompleto_nuevo1 = []
    imagenes = []
    for i in range(0, arrayCompleto.shape[0]):
        imagen = arrayCompleto[i]
        W, H = imagen.shape[:2]
        new_W, new_H = (w, h)
        xrange = lambda x: np.linspace(0, 1, x)
        
        f = interpolate.interp2d(xrange(W), xrange(H), imagen, kind="linear")
        new_arr = f(xrange(new_W), xrange(new_H))
        new_arr_1 = resize(new_arr, (w, h, 1))
        arrayCompleto_nuevo.append(new_arr_1)#este resize lo hacemos luego
        arrayCompleto_nuevo1.append(new_arr)

    return np.asarray(arrayCompleto_nuevo1)


def plot1(x):
    for i in range(numImages):
    # define subplot
        pyplot.subplot(330 + 1 + i)
        # plot raw pixel data
        pyplot.imshow(x[i+20])
        
    # show the figure
    pyplot.show()

def plot(x, y_train):
    cont = 1
    fila = 5
    col = 5
    f, axarr = plt.subplots(fila, col)

    for i in range(fila):
        for j in range(col):
            axarr[i, j].imshow(x[cont])
            axarr[i, j].set_title("Objeto "+str(y_train[cont]))
            axarr[i, j].axis('off')
            cont = cont + 1

    '''i = i+1
    axarr[0,1].imshow(x[i])
    axarr[0,1].set_title(y_train[i])
    axarr[0,1].axis('off')

    i = i+1
    axarr[1,0].imshow(x[i])
    axarr[1,0].set_title(y_train[i])
    axarr[1,0].axis('off')

    i = i+1
    axarr[1,1].imshow(x[i])
    axarr[1,1].set_title(y_train[i])
    axarr[1,1].axis('off')'''

    plt.show()


def contructDataframe(file_list):
    """
    this function builds a data frame which contains 
    the path to image and the tag/object name using the prefix of the image name
    """
    x_train=[]
    y_train=[]

    for f in file_list:
    	#print(os.path.join(path + f))
    	x_train.append( cv2.imread(os.path.join(path + f)) )
    	label = os.path.splitext(f)[0]
    	end = label.find('__', 0)
    	label_int=int(label[3:end])
    	y_train.append(label_int)

    return np.asarray(x_train), y_train

cont = 0
for f in files1:    
    print(os.path.splitext(f)[0])
    if (os.path.splitext(f)[1] == ".png"):
        print(os.path.splitext(f)[1])
    cont = cont + 1
print("cont: ", cont)
x_train, y_train = contructDataframe(files1)
#print("shape x_train: ", x_train.shape, "shape y_train: ", y_train.shape)

#print("convert to rgb2gray...")
#x_train = other_utils.rgb2gray(x_train)
#print(x_train.shape)
plot(x_train, y_train)


#print("resize28...")
#x_train = resize28(x_train, 28, 28)
#print(x_train.shape)
#plot(x_train)
y = y_train
print(type(y))
print(y)
#data = [1, 3, 2, 0, 3, 2, 2, 1, 0, 1]
#print(type(data))
# one hot encode
encoded = to_categorical(y)
print(encoded.shape)
print("encoded: ", encoded[0])
# invert encoding
y_train = argmax(encoded[0])

print("y_train: ", y_train)


#X_train, X_test, y_train, y_test = train_test_split(df.path, df.label, test_size=0.20,random_state=0,stratify= df.label)

#X_train=[img_to_array(cv2.imread(file).astype("float")/255.0) for file in tqdm(X_train.values)]
#X_test=[img_to_array(cv2.imread(file).astype("float")/255.0) for file in tqdm(X_test.values)]

#print(len(X_train))
#img = X_train[0]
#plt.imshow(img)
#plt.show()

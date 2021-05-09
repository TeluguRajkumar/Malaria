#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 11:47:03 2021

@author: rajkumar
"""

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
print(os.listdir('../input'))
#The next hidden code cells define functions for plotting data
# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()
    
# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()
    
# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()
    
    
    
######malaria cell image using Deep laerining######

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from PIL import Image


def load_images_from_folder(folder,id,l,w):
    
    for filename in os.listdir(folder):
        if filename!="Thumbs.db":
            img=cv2.imread(os.path.join(folder,filename))
            #print(img.shape)
            if img.shape[0]<l:
                    l=img.shape[0]
            if img.shape[1]<w:
                    w=img.shape[1]
    return l,w
                    
l,w=load_images_from_folder("../input/cell_images/cell_images/Parasitized/",2,1000000,1000000)
l,w=load_images_from_folder("../input/cell_images/cell_images/Uninfected/",2,l,w)
print(l,w)
#pip install python-resize-image
from resizeimage import resizeimage
images=[]
labels=[]
def load_images_from_folder_a(folder,id):
    
    for filename in os.listdir(folder):
        if filename!="Thumbs.db":
            img1 = Image.open(os.path.join(folder,filename))
            new1 = resizeimage.resize_contain(img1, [40, 46, 3])
            new1 = np.array(new1, dtype='uint8')
            images.append(new1)
            if id==1:
                labels.append(0)
            else:
                labels.append(1)
                
load_images_from_folder_a("../input/cell_images/cell_images/Parasitized",1)
load_images_from_folder_a("../input/cell_images/cell_images/Uninfected/",2)
print(len(images))
print(len(labels))
train = np.array(images)
label=labels
train = train.astype('float32') / 255
if label[1]==1:
    plt.title("Parasitized")
    plt.imshow(train[1])
else:
    plt.title("Uninfected")
    plt.imshow(train[1])

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(train,label,test_size=0.2,random_state=1)
import keras
from keras import Sequential, utils
print(len(X_train),len(X_test))
print(len(Y_train),len(Y_test))
print(X_train[100].shape)
#Doing One hot encoding as classifier has multiple classes
Y_train=keras.utils.to_categorical(Y_train,2)
Y_test=keras.utils.to_categorical(Y_test,2)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding
from keras.layers import SpatialDropout1D, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(46, 40, 3))) 
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
    
model.add(Dense(2, activation='softmax'))

model.summary() 

model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=["accuracy"])
plt.title(Y_train[0])
plt.imshow(X_train[0])
model.fit(X_train,Y_train, epochs=10, batch_size=52, shuffle=True, validation_data=(X_test,Y_test))
accuracy = model.evaluate(X_test, Y_test, verbose=1)
print('\n', 'Test_Accuracy:-', accuracy[1])
from keras.models import load_model
model.save('cells.h5')





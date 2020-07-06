import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.metrics import accuracy_score

num_class = 43
img_rows, img_cols = 30, 30

path = os.getcwd()
path_train = os.path.join(path, 'train')
path_test = os.path.join(path,'test')

x_train = []
y_train = []


for i in range(num_class) :
    path_i = os.path.join(path_train,str(i))
    images = os.listdir(path_i)
    for j in images:
       img = Image.open(os.path.join(path_i,str(j)))
       #img = img.resize((30,30))
       arr = np.asarray(img, dtype="float32")
       x_train.append(arr)
       y_train.append(i)
       
x_train = np.array(x_train)
y_train = np.array(y_train)

def model_CNN():
       model = Sequential()

       model.add(Conv2D(32, (3, 3),input_shape=( img_rows, img_cols ,3),activation='relu'))
       model.add(Conv2D(32, (3, 3), activation='relu'))
       model.add(MaxPool2D(pool_size=(2, 2)))
       model.add(Dropout(0.2))

       model.add(Conv2D(64, (3, 3),activation='relu'))
       model.add(Conv2D(64, (3, 3), activation='relu'))
       model.add(MaxPool2D(pool_size=(2, 2)))
       model.add(Dropout(0.2))

       model.add(Conv2D(128, (3, 3),activation='relu'))
       model.add(Conv2D(128, (3, 3), activation='relu'))
       model.add(MaxPool2D(pool_size=(2, 2)))
       model.add(Dropout(0.2))

       model.add(Flatten())
       model.add(Dense(512, activation='relu'))
       model.add(Dropout(0.5))
       model.add(Dense(num_class, activation='softmax'))

       model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
       return model


X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
Y_train = to_categorical(Y_train, num_class)
Y_test = to_categorical (Y_test,num_class)

batchsize = 32
epochs = 10
model = model_CNN()
train = model.fit(X_train, Y_train, batch_size=batchsize, epoch = epochs , validation_data=(X_test,Y_test) )

test = pd.read_csv('Test.csv')
y = test["ClassId"].values
imgs = test["Path"].values
data=[]
for img in imgs:
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))
X=np.array(data)
y_pred = model.predict_classes(X)
#Accuracy with the test data
from sklearn.metrics import accuracy_score
print(accuracy_score(y, y_pred))

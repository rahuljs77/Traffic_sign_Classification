from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pickle
import random
import glob
import csv
import cv2
import os

#  Loading Data
training_file = 'traffic_signs/train.p' # pickle data file directory
validation_file = 'traffic_signs/valid.p'
testing_file = 'traffic_signs/test.p'

print('loading data...')
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    validation = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

x_train, y_train = train['features'], train['labels']
x_validation, y_validation = validation['features'], validation['labels']
x_test, y_test = test['features'], test['labels']

label_binarizer = LabelBinarizer()
y_hot_train = label_binarizer.fit_transform(y_train)
y_hot_valid = label_binarizer.fit_transform(y_validation)
y_hot_test = label_binarizer.fit_transform(y_test)

print()
print('completed loading data')
print(len(x_train))

# labeling the images based on their index
labels = []
with open('signnames.csv', 'r') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if row[1] == 'SignName':
            pass
        else:
            labels.append(row[1])

print("Completed labelling data")

#  Data visualisation:for i in range(0, len(X_train)):
label_check = []
for i in range(0, len(x_train)):
    image = (x_train[i])
    label = y_train[i]
    if label in label_check:
        pass
    else:
        label_check.append(label)
        plt.imshow(image)
        plt.title(labels[label])
#         plt.show()

print("normalizing data...")

image_shape = x_train[0].shape


def Normalize(image_set):
    new_shape = image_shape[0:2] + (1,)

    prep_image_set = np.empty(shape=(len(image_set),) + new_shape, dtype=int)

    for ind in range(0, len(image_set)):
        norm_img = cv2.normalize(image_set[ind], np.zeros(image_shape[0:2]), 0, 255, cv2.NORM_MINMAX)
        norm_img = norm_img.astype(np.float32)

        # grayscale
        gray_img = cv2.cvtColor(norm_img, cv2.COLOR_BGR2GRAY)
        prep_image_set[ind] = np.reshape(gray_img, new_shape)

    return prep_image_set


X_train = Normalize(x_train)
X_valid = Normalize(x_validation)
X_test = Normalize(x_test)

print("Normalization done!")

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(32, 32, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(48, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dropout(0.7))
model.add(Dense(300))
model.add(Activation('relu'))
model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dense(43))
model.add(Activation('softmax'))

model.compile('adam', 'categorical_crossentropy', ['accuracy'])
model.fit(X_train, y_hot_train, batch_size=128, epochs=15, shuffle=True, validation_data=(X_valid, y_hot_valid), verbose=2)






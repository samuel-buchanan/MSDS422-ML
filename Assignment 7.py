
import pandas as pd
import matplotlib.pyplot as plt
import random
from matplotlib import image
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.decomposition import PCA
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from keras import layers, models, optimizers
import os

# 12500 images each of cats and dogs (0-12499)
# let's select 2500 at random to use for a reduced training set
selected_list = random.sample(range(12500), 2500)

# get filenames of matching cat and dog selected_list numbers
cat_train_list = []
dog_train_list = []
for i in selected_list:
    cat_filename = "cat." + str(i) + ".jpg"
    dog_filename = "dog." + str(i) + ".jpg"
    cat_train_list.append(cat_filename)
    dog_train_list.append(dog_filename)

# use filename lists to provide which images to resize

x = [] # images as arrays
y = [] # labels, 0 = cat, 1 = dog

for image in cat_train_list:
    x.append(cv2.resize(cv2.imread("train/" + image), (200, 200), interpolation=cv2.INTER_CUBIC))
for image in dog_train_list:
    x.append(cv2.resize(cv2.imread("train/" + image), (200, 200), interpolation=cv2.INTER_CUBIC))
    
# first 2500 are cat, rest are dog
# I'm sure there's an easier way, but I was tired when I wrote this
for i in range(2500):
    y.append(0)
for i in range(2500):
    y.append(1)

# train / test split for cross-validation
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)


# define model layers
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), input_shape=(200, 200, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1))
model.add(layers.Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


x = np.array(x)
x_train = np.array(x_train)
model.fit(x, y, epochs=10, validation_split = 0.2)
# optimizer = adam, dropout = 0.2
# 0.9360 train accuracy, 0.3180 val accuracy

# prepare test data for prediction
test_list = os.listdir("test/")
x_pred = []
for image in test_list:
    x_pred.append(cv2.resize(cv2.imread("test/" + image), (200, 200), interpolation=cv2.INTER_CUBIC))

# make predictions
x_pred = np.array(x_pred)
adam_02 = model.predict_classes(x_pred)

# generate list of ID's to match with predictions
ID_list = []
for element in test_list:
    ID = element.split('.')
    ID_list.append(ID[0])

# put it all together and what do you get
adam_02_2 = list(adam_02)
for i in range(len(adam_02_2)):
    word = str(adam_02_2[i])
    word = word.lstrip('[')
    word = word.rstrip(']')
    adam_02_2[i] = word

adam_02_df = pd.DataFrame({'id' : ID_list, 'label' : adam_02_2})
adam_02_df['id'] = pd.to_numeric(adam_02_df['id'])
adam_02_df = adam_02_df.sort_values('id')
adam_02_df.to_csv('adam_02_predictions.csv', index = False)



# model2: optimizer = adam, dropout = 0.5
model2 = models.Sequential()

model2.add(layers.Conv2D(32, (3, 3), input_shape=(200, 200, 3)))
model2.add(layers.Activation('relu'))
model2.add(layers.MaxPooling2D(pool_size=(2, 2)))

model2.add(layers.Conv2D(32, (3, 3)))
model2.add(layers.Activation('relu'))
model2.add(layers.MaxPooling2D(pool_size=(2, 2)))

model2.add(layers.Conv2D(64, (3, 3)))
model2.add(layers.Activation('relu'))
model2.add(layers.MaxPooling2D(pool_size=(2, 2)))

model2.add(layers.Flatten())
model2.add(layers.Dense(64))
model2.add(layers.Activation('relu'))
model2.add(layers.Dropout(0.5))
model2.add(layers.Dense(1))
model2.add(layers.Activation('sigmoid'))

model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model2.fit(x, y, epochs=10, validation_split = 0.2)
# train accuracy = 0.6250, val accuracy = ~0
adam_05 = model2.predict_classes(x_pred)
adam_05_2 = list(adam_05)
for i in range(len(adam_05_2)):
    word = str(adam_05_2[i])
    word = word.lstrip('[')
    word = word.rstrip(']')
    adam_05_2[i] = word

adam_05_df = pd.DataFrame({'id' : ID_list, 'label' : adam_05_2})
adam_05_df['id'] = pd.to_numeric(adam_05_df['id'])
adam_05_df = adam_05_df.sort_values('id')
adam_05_df.to_csv('adam_05_predictions.csv', index = False)


# model3: optimizer = rmsprop, droput = 0.2
model3 = models.Sequential()

model3.add(layers.Conv2D(32, (3, 3), input_shape=(200, 200, 3)))
model3.add(layers.Activation('relu'))
model3.add(layers.MaxPooling2D(pool_size=(2, 2)))

model3.add(layers.Conv2D(32, (3, 3)))
model3.add(layers.Activation('relu'))
model3.add(layers.MaxPooling2D(pool_size=(2, 2)))

model3.add(layers.Conv2D(64, (3, 3)))
model3.add(layers.Activation('relu'))
model3.add(layers.MaxPooling2D(pool_size=(2, 2)))

model3.add(layers.Flatten())
model3.add(layers.Dense(64))
model3.add(layers.Activation('relu'))
model3.add(layers.Dropout(0.2))
model3.add(layers.Dense(1))
model3.add(layers.Activation('sigmoid'))

model3.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# train accuracy = 0.6250, val accuracy = ~0
model3.fit(x, y, epochs=10, validation_split = 0.2)
# 
rmsprop_02 = model3.predict_classes(x_pred)
rmsprop_02_2 = list(rmsprop_02)
for i in range(len(rmsprop_02_2)):
    word = str(rmsprop_02[i])
    word = word.lstrip('[')
    word = word.rstrip(']')
    rmsprop_02_2[i] = word

rmsprop_02_df = pd.DataFrame({'id' : ID_list, 'label' : rmsprop_02_2})
rmsprop_02_df['id'] = pd.to_numeric(rmsprop_02_df['id'])
rmsprop_02_df = rmsprop_02_df.sort_values('id')
rmsprop_02_df.to_csv('rmsprop_02_predictions.csv', index = False)


# model4: optimizer = rmsprop, droput = 0.5
model4 = models.Sequential()

model4.add(layers.Conv2D(32, (3, 3), input_shape=(200, 200, 3)))
model4.add(layers.Activation('relu'))
model4.add(layers.MaxPooling2D(pool_size=(2, 2)))

model4.add(layers.Conv2D(32, (3, 3)))
model4.add(layers.Activation('relu'))
model4.add(layers.MaxPooling2D(pool_size=(2, 2)))

model4.add(layers.Conv2D(64, (3, 3)))
model4.add(layers.Activation('relu'))
model4.add(layers.MaxPooling2D(pool_size=(2, 2)))

model4.add(layers.Flatten())
model4.add(layers.Dense(64))
model4.add(layers.Activation('relu'))
model4.add(layers.Dropout(0.5))
model4.add(layers.Dense(1))
model4.add(layers.Activation('sigmoid'))

model4.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model4.fit(x, y, epochs=10, validation_split = 0.2)
# train accuracy = 0.6250, val accuracy = ~0
rmsprop_05 = model4.predict_classes(x_pred)
rmsprop_05_2 = list(rmsprop_05)
for i in range(len(rmsprop_05_2)):
    word = str(rmsprop_05[i])
    word = word.lstrip('[')
    word = word.rstrip(']')
    rmsprop_05_2[i] = word

rmsprop_05_df = pd.DataFrame({'id' : ID_list, 'label' : rmsprop_05_2})
rmsprop_05_df['id'] = pd.to_numeric(rmsprop_05_df['id'])
rmsprop_05_df = rmsprop_05_df.sort_values('id')
rmsprop_05_df.to_csv('rmsprop_05_predictions.csv', index = False)


# model5: optimizer = adam, droput = 0.01
model5 = models.Sequential()

model5.add(layers.Conv2D(32, (3, 3), input_shape=(200, 200, 3)))
model5.add(layers.Activation('relu'))
model5.add(layers.MaxPooling2D(pool_size=(2, 2)))

model5.add(layers.Conv2D(32, (3, 3)))
model5.add(layers.Activation('relu'))
model5.add(layers.MaxPooling2D(pool_size=(2, 2)))

model5.add(layers.Conv2D(64, (3, 3)))
model5.add(layers.Activation('relu'))
model5.add(layers.MaxPooling2D(pool_size=(2, 2)))

model5.add(layers.Flatten())
model5.add(layers.Dense(64))
model5.add(layers.Activation('relu'))
model5.add(layers.Dropout(0.01))
model5.add(layers.Dense(1))
model5.add(layers.Activation('sigmoid'))

model5.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model5.fit(x, y, epochs=10)
# train accuracy = 0.9772
adam_001 = model5.predict_classes(x_pred)
adam_001_2 = list(adam_001)
for i in range(len(adam_001_2)):
    word = str(adam_001[i])
    word = word.lstrip('[')
    word = word.rstrip(']')
    adam_001_2[i] = word

adam_001_df = pd.DataFrame({'id' : ID_list, 'label' : adam_001_2})
adam_001_df['id'] = pd.to_numeric(adam_001_df['id'])
adam_001_df = adam_001_df.sort_values('id')
adam_001_df.to_csv('adam_001_predictions.csv', index = False)
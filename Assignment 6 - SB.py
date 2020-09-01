
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import tensorflow as tf


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

label = train.label

train.drop("label", axis = 1, inplace = True)

# have to convert data from integers to floats for TF
train = train/255.0

train_dataset = train.values.reshape(42000, 28, 28, 1)



model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), input_shape = (28,28,1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(10, activation="softmax")
        ])

    
model.compile(optimizer='adam',
              loss = "sparse_categorical_crossentropy",
              metrics=['accuracy'])


start = datetime.now()
model.fit(train_dataset, label, epochs=5)
end = datetime.now()
print(end-start)
# 5 epochs, dropout 0.1: accuracy 0.9247, time 1min13.5168sec


model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), input_shape = (28,28,1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(10, activation="softmax")
        ])

    
model.compile(optimizer='adam',
              loss = "sparse_categorical_crossentropy",
              metrics=['accuracy'])


start = datetime.now()
model.fit(train_dataset, label, epochs=10)
end = datetime.now()
print(end-start)
# 10 epochs, dropout 0.1: accuracy 0.9284, time 2min28.3528sec


model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), input_shape = (28,28,1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax")
        ])

    
model.compile(optimizer='adam',
              loss = "sparse_categorical_crossentropy",
              metrics=['accuracy'])


start = datetime.now()
model.fit(train_dataset, label, epochs=5)
end = datetime.now()
print(end-start)
# 5 epochs, dropout 0.2: accuracy 0.9229, time 1min14.6872sec


model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), input_shape = (28,28,1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax")
        ])

    
model.compile(optimizer='adam',
              loss = "sparse_categorical_crossentropy",
              metrics=['accuracy'])


start = datetime.now()
model.fit(train_dataset, label, epochs=10)
end = datetime.now()
print(end-start)
# 10 epochs, dropout 0.2: accuracy 0.9285, time 2min36.1074sec


# predictions
# train_dataset = train.values.reshape(42000, 28, 28, 1)
test = test/255.0
test_dataset = test.values.reshape(28000, 28, 28, 1)
test_predictions = model.predict(test_dataset)

results = np.argmax(test_predictions,axis = 1)
results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("max_acc_NN_submission.csv",index=False)


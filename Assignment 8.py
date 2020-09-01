import pandas as pd
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import LSTM, GRU
from keras.layers import Dense, Embedding, Bidirectional, Dropout, Flatten
from keras.optimizers import Adam, SGD
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

train_txt = train['text']
test_txt = test['text']
all_txt = train_txt.append(test_txt)

# let's make our vocabulary for tokenization with all the tweets
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_txt)
len(tokenizer.word_index)

# dropping other columns, looking only at text
train = train.drop(['id', 'keyword', 'location'], axis = 1)
test = test.drop(['id', 'keyword', 'location'], axis = 1)

# make list of labels for later use
labels = train['target']
train = train.drop(['target'], axis = 1)

# tokenize!
train_token = tokenizer.texts_to_sequences(train.text)
test_token = tokenizer.texts_to_sequences(test.text)

# padding short tweets to get everything the same size
max_length = max(len(x.split()) for x in all_txt)
train_pad = pad_sequences(train_token, maxlen=max_length, padding='post')
test_pad = pad_sequences(test_token, maxlen=max_length, padding='post')

# before and after comparison
train.text[423]
train_token[423]
train_pad[423]
# looks pretty good


# going to try a LSTM vs GRU model, here's model1
model1 = Sequential()
model1.add(Embedding(len(tokenizer.word_index) + 1, 100, input_length = max_length))
model1.add(Bidirectional(LSTM(128)))
model1.add(Dropout(0.2))
model1.add(Dense(256, activation='relu'))
model1.add(Dropout(0.2))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# model2, using GRU
model2 = Sequential()
model2.add(Embedding(len(tokenizer.word_index) + 1, 100, input_length = max_length))
model2.add(Bidirectional(GRU(128)))
model2.add(Dropout(0.2))
model2.add(Dense(256, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# fitting models
model1_fit = model1.fit(train_pad, labels, epochs = 5, validation_split = 0.2)
    # accuracy = 0.9923, val_accuracy = 0.7203
model2_fit = model2.fit(train_pad, labels, epochs = 5, validation_split = 0.2)
    # accuracy = 0.9938, val_accuracy = 0.7321
    
# make predictions
LSTM_pred = model1.predict(test_pad)
GRU_pred = model2.predict(test_pad)

# convert to binary
LSTM_pred_final = np.empty(len(LSTM_pred), dtype = int)
for i in range(len(LSTM_pred)):
    if LSTM_pred[i] > 0.5:
        LSTM_pred_final[i] = 1
    else:
        LSTM_pred_final[i] = 0

GRU_pred_final = np.empty(len(GRU_pred), dtype = int)
for i in range(len(GRU_pred)):
    if GRU_pred[i] > 0.5:
        GRU_pred_final[i] = 1
    else:
        GRU_pred_final[i] = 0


# package for Kaggle
LSTM_pred_file = pd.read_csv('sample_submission.csv')
LSTM_pred_file.target = LSTM_pred_final
LSTM_pred_file.to_csv('LSTM_pred.csv', index = False)

LSTM_pred_file.target = GRU_pred_final
LSTM_pred_file.to_csv('GRU_pred.csv', index = False)



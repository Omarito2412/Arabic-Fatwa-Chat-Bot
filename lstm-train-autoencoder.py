import pandas as pd
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, RepeatVector
from keras.utils import np_utils
from nltk.stem import ISRIStemmer
from six.moves import cPickle

BATCH_SIZE = 32 # Batch size for GPU
NUM_WORDS = 10000 # Vocab length
MAX_LEN = 20 # Padding length (# of words)
LSTM_EMBED = 8 # Number of LSTM nodes

def batches_generator(train_data, batch_size=32):
    # For OHE inputs
    num_words = np.max(train_data) + 1
    timesteps = train_data.shape[1]
    while True:
        indices = np.random.choice(len(train_data), size=batch_size)
        X = train_data[indices]
        X = np_utils.to_categorical(X, num_words)
        X = X.reshape((batch_size, timesteps, num_words))
        yield (X, X)

train_data = pd.read_csv("/home/omar/DataScience/DataSets/askfm/full_dataset.csv")

stemmer = ISRIStemmer()

# We don't need the answers, so let's drop them
train_data.drop('Answer', inplace=True, axis=1)

train_data = train_data[train_data.Question.apply(lambda x: len(x.split())) < MAX_LEN]

train_data.Question = train_data.Question.apply(lambda x: (re.sub('[^\u0620-\uFEF0\s]', '', x)).strip())

train_data = train_data[train_data.Question.apply(len) > 0]

# Stem the words
train_data.Question = train_data.Question.apply(lambda x: " ".join([stemmer.stem(i) for i in x.split()]))


tokenizer = Tokenizer(num_words=NUM_WORDS, lower=False)

tokenizer.fit_on_texts(train_data["Question"].values)

# Save the tokenizer for later use
cPickle.dump(tokenizer, open("models/lstm-autoencoder-tokenizer.pickle", "wb"))

train_data = tokenizer.texts_to_sequences(train_data["Question"].values)

train_data = pad_sequences(train_data, padding='post', truncating='post', maxlen=MAX_LEN)

model = Sequential()
model.add(Embedding(NUM_WORDS, 100, input_length=MAX_LEN))
model.add(LSTM(LSTM_EMBED, dropout=0.2, recurrent_dropout=0.2, input_shape=(train_data.shape[1], NUM_WORDS)))
model.add(RepeatVector(train_data.shape[-1]))
model.add(LSTM(LSTM_EMBED, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(Dense(NUM_WORDS, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
# model.fit_generator(batches_generator(train_data), steps_per_epoch=(len(train_data) // BATCH_SIZE))
model.fit(train_data, np.expand_dims(train_data, -1), epochs=25, batch_size=BATCH_SIZE)

model.save("models/lstm-encoder.h5")

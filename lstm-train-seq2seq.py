import pandas as pd
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding
from keras.utils import np_utils
import seq2seq
from seq2seq.models import AttentionSeq2Seq
from nltk.stem import ISRIStemmer
from six.moves import cPickle

BATCH_SIZE = 32
NUM_WORDS = 1000
MAX_LEN = 25

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

stemmer = ISRIStemmer()

data = pd.read_csv("/home/omar/DataScience/DataSets/askfm/full_dataset.csv")

data = data[data.Answer.apply(lambda x: len(x.split())) < MAX_LEN]
data = data[data.Question.apply(lambda x: len(x.split())) < MAX_LEN]

data.Question = data.Question.apply(lambda x: (re.sub('[^\u0620-\uFEF0\s]', '', x)).strip())
data.Answer = data.Answer.apply(lambda x: (re.sub('[^\u0620-\uFEF0\s]', '', x)).strip())

data = data[data.Answer.apply(len) > 0]
data = data[data.Question.apply(len) > 0]

data.Question = data.Question.apply(lambda x: " ".join([stemmer.stem(i) for i in x.split()]))
data.Answer = data.Answer.apply(lambda x: " ".join([stemmer.stem(i) for i in x.split()]))

tokenizer = Tokenizer(num_words=NUM_WORDS, lower=False)

train_data = pd.concat((data.Question, data.Answer), ignore_index=True)
tokenizer.fit_on_texts(train_data)
cPickle.dump(tokenizer, open("models/lstm-seq2seq-tokenizer.pickle", "wb"))

Questions = tokenizer.texts_to_sequences(data.Question)
Answers = tokenizer.texts_to_sequences(data.Answer)

Questions = pad_sequences(Questions, padding='post', truncating='post', maxlen=MAX_LEN)
Answers = pad_sequences(Answers, padding='post', truncating='post', maxlen=MAX_LEN)

model = Sequential()
model.add(Embedding(NUM_WORDS, 200, input_length=MAX_LEN))
attn = AttentionSeq2Seq(batch_input_shape=(None, MAX_LEN, 200), hidden_dim=10, output_length=MAX_LEN, output_dim=NUM_WORDS, depth=1)
model.add(attn)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
model.fit(Questions, np.expand_dims(Answers, 2), batch_size=BATCH_SIZE, epochs=10)
model.save("models/lstm-seq2seq.h5")

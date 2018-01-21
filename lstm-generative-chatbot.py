from keras.models import load_model, Sequential
from keras.layers import Embedding
from six.moves import cPickle
import numpy as np
import seq2seq
from seq2seq.models import AttentionSeq2Seq
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

BATCH_SIZE = 32
NUM_WORDS = 1000
MAX_LEN = 25

tokenizer = cPickle.load(open("models/lstm-seq2seq-tokenizer.pickle", "rb"))
model = Sequential()
model.add(Embedding(NUM_WORDS, 200, input_length=MAX_LEN))
attn = AttentionSeq2Seq(batch_input_shape=(None, MAX_LEN, 200), hidden_dim=10, output_length=MAX_LEN, output_dim=NUM_WORDS, depth=1)
model.add(attn)
model.load_weights("models/lstm-seq2seq.h5")

index2word = dict([(v, k) for (k,v) in tokenizer.word_index.items()])
index2word[0] = "PAD"

while True:
    question = [input('Please enter a question: \n')]
    question = tokenizer.texts_to_sequences(question)
    question = pad_sequences(question, padding='post', truncating='post', maxlen=MAX_LEN)
    answer = np.squeeze(model.predict_classes(question))
    answer = [index2word[item] for item in answer]
    print(" ".join(answer))


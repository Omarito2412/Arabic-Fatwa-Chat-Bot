import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


data = pd.read_csv("/home/omar/DataScience/DataSets/askfm/full_dataset.csv")
vectorizer = TfidfVectorizer()

vectorizer.fit(data.values.ravel())

while True:
    question = [input('Please enter a question: \n')]
    question = vectorizer.transform(question)

    rank = cosine_similarity(question, vectorizer
                             .transform(data['Question'].values))
    top = np.argsort(rank, axis=-1).T[-5:].tolist()
    for item in top:
        print(data['Answer'].iloc[item].values[0])
        print("\n ########## \n")

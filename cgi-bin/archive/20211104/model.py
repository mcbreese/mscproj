#!/usr/bin/env python3
# ^ Require the above to point to Python.exe - LEAVE IT IN

# Use to access the csv
import pandas as pd
# Use to access the data in the csv as arrays
import numpy as np
# Need to vectorize text
from sklearn.feature_extraction.text import CountVectorizer
# Need to train and test model
from sklearn.model_selection import train_test_split
# The training model
from sklearn.naive_bayes import MultinomialNB
import pickle

# Same folder as this .py script
data = pd.read_csv('C:\\Users\\thoma\\Documents\\test\\data\\fake_or_real_news.csv')
# Assign value x to the text body
x = np.array(data["text"])
# Assign value y to the label (fake or real)
y = np.array(data["label"])

# Count vectorizer is a built in sklearn function which converts text to a matrix of token souncts
cv = CountVectorizer()
# Learn the vocabulary dictionary and return document-term matrix, only do it to text and not label classifier
x = cv.fit_transform(x)

# Seperate into test and training sets
# test size 20%
# Random state ensures that the splits that you generate are reproducible
# 42 also the answer to life, the universe and everything
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model using multinomial naive bayes algorithm (uses bag or words)
model = MultinomialNB()
model.fit(xtrain, ytrain)
# Output is 0.89 what does that mean??? -- Accuracy??
#print(model.score(xtest, ytest))

#print ("Model trained. Saving model to model.pickle")
with open("model.pickle", "wb") as file:
    pickle.dump(model, file)
#print('Model saved')

# Call the vectorizer function from main.py, only works when cv has been used in training
# Need to vectorize seperately without running the whole script really
def vectorize_page(string):
        data = cv.transform([string]).toarray()
        return data

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
# Scrapes page text
from test import return_text
import cgi


print("Content-Type: text/html\n\r\n")
form = cgi.FieldStorage()
# Get data from fields
url = form.getvalue('url')

# Get data from fields
url = form.getvalue('url')

# Same folder as this .py script
data = pd.read_csv("../data/fake_or_real_news.csv")
print(data.head())
# Assign value x to the text body
x = np.array(data["text"])
# Assign value y to the label (fake or real)
y = np.array(data["label"])

# Count vectorizer is a built in sklearn function which converts text to a matrix of token souncts
cv = CountVectorizer()
# Learn the vocabulary dictionary and return document-term matrix
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
print(model.score(xtest, ytest))


# This will be the scraped text below
url='https://towardsdatascience.com/machine-learning-explainability-introduction-via-eli5-99c767f017e2'
print(url)
page_text=return_text(url)
print(page_text)
# Transorm to vectors
data = cv.transform([page_text]).toarray()
print(model.predict(data))
# use predict feature on the vectorized data

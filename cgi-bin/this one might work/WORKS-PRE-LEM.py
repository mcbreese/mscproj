#!/usr/bin/env python3
# ^ Require the above to point to Python.exe - LEAVE IT IN

# Use to access the csv
import pandas as pd
# Use to access the data in the csv as arrays
import numpy as np
# Need to vectorize text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
# Need to train and test model
from sklearn.model_selection import train_test_split
# The training model
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from ml_helper.helper import Helper
from sklearn.metrics import accuracy_score as metric_scorer
from sklearn.metrics import classification_report, confusion_matrix
import pickle

KEYS = {
    "SEED": 1,
    "DATA_PATH": "C:\\Users\\thoma\\Documents\\test\\data\\fake_or_real_news.csv",
    "TARGET": "label",
    "METRIC": "accuracy",
    "TIMESERIES": False,
    "SPLITS": 3,
    "ESTIMATORS": 150,
    "ITERATIONS": 500,
}
# Simplify helper object into variable
hp = Helper(KEYS)
# ds = dataset, read the CSV in the Keys
ds = pd.read_csv('C:\\Users\\thoma\\Documents\\test\\data\\fake_or_real_news.csv')
#ds = pd.read_csv(KEYS["DATA_PATH"], header=0, names=["id", "title", "text", "label"])
# Merge together the two columns header and title
merge = np.array(ds["title"] + ds["text"])
# Create a label variable which is all of the labels in the dataset fake or real
label= np.array(ds["label"])

#-------------------------------------
# PRE-PROCESS OR IMPORT DEF
#-------------------------------------


#-------------------------------------
# 4. Vectorising the data
#-------------------------------------
# Count vectorizer is a built in sklearn function which converts text to a matrix of token souncts
cv = CountVectorizer()
# Learn the vocabulary dictionary and return document-term matrix, only do it to text and not label classifier
# All the text columns are now floats
x = cv.fit_transform(merge)
tfidf = TfidfTransformer()
x = tfidf.fit_transform(x)


# Create a train and test ds variables, the test size is 20% and train 80%, randomise using the seed but is constant to make it reproducible
xtrain, xtest, ytrain, ytest = train_test_split(x, label, test_size=0.20, random_state=KEYS["SEED"])

# Train model using multinomial naive bayes algorithm (uses bag or words)
model = PassiveAggressiveClassifier(max_iter=KEYS["ITERATIONS"], random_state=KEYS["SEED"], tol=1e-3)
# Find out what this is actually doing
model.fit(xtrain, ytrain)
# Accuracy output using the 2 testing sets - same as metric scorer?
print(model.score(xtest, ytest))

#print(xtrain)
#print(model.predict(ds["merge"]))
#xpredictions=model.predict(xtest)
#ypredictions=model.predict(ytest)
#print(xpredictions)
#print(metric_scorer(xpredictions))
#print(classification_report(xtest))

#print ("Model trained. Saving model to model.pickle")
with open("idk.pickle", "wb") as file:
    pickle.dump(model, file)
print('Model saved')

# Call the vectorizer function from main.py, only works when cv has been used in training
# Need to vectorize seperately without running the whole script really
def vectorize_page(string):
        data = cv.transform([string]).toarray()
        data = tfidf.transform(data)
        return data

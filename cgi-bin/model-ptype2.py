#!/usr/bin/env python3
# ^ Require the above to point to Python.exe - LEAVE IT IN

# Regular expressions library
import re
# USED TO EXPLAIN THE MODEL
import eli5
# spaCy is for NLP
import spacy
# Natural language tool kit
import nltk as nl
# Pandas is for opening files e.g. CSVs
import pandas as pd
from sklearn.base import clone
# Helpers to speed up and structure machine learning projects with KEYS
import matplotlib.pyplot as plt
from scipy.sparse import hstack
# Import stop words to remove from our text
from nltk.corpus import stopwords
from ml_helper.helper import Helper
# Import the below models
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#from scikitplot.metrics import plot_confusion_matrix
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score as metric_scorer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
# Text pre-processing languages
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer, WordNetLemmatizer
#Only need the below to update the list of stopwords from nltk
#nl.download('stopwords')
#%matplotlib inline

# Need to understand what all of this is
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

hp = Helper(KEYS)

# ds = dataset, read the CSV in the Keys
# !!!!!Change some of the names
ds = pd.read_csv(KEYS["DATA_PATH"], header=0, names=["id", "title", "text", "label"])
# Create a train and test ds variables, the test size is 20% and train 80%, randomise using the seed
train, test = train_test_split(ds, test_size=0.20, random_state=KEYS["SEED"])

# Show the datatypes of the train dataset
#print(train.dtypes)
# Print the head of the dataset
#print(train.head)
# Check for missing data using the helper library built in function
#print(hp.missing_data(ds))
# Print the location row 10 column 2 of training dataset (seed has changed order), using iloc (integer location) from pandas
#print(train.iloc[1,2])

# Merge the title and the body of the text from the dataset being used
train["merge"] = train["title"] + train["text"]
test["merge"] = test["title"] + test["text"]
#print(train.head())

# Count vectorizer is a built in sklearn function which converts text to a matrix of token souncts
cv = CountVectorizer()
# This is the exact same as prototype 1, vectorize the training set to convert articles to matrix of token counts
train_data = cv.fit_transform(train["merge"])
#print(train_data.shape)
# Transform matrix of token counts (from cv) to tf-idf representation - importance of repeated tokens throughout the text is reduced
tfidf = TfidfTransformer()
train_data = tfidf.fit_transform(train_data)
print(train_data.shape)

#------------------------------------------------------------------------------------------------
# Principle Component Analysis not done here but COULD BE SPOKEN ABOUT IN THE BODY OF THE ESSAY?
#------------------------------------------------------------------------------------------------

# Testing of models to see which one is the best

basepipe = Pipeline([
    ('vect', TfidfVectorizer(stop_words="english", ngram_range=(1,2), sublinear_tf=True))
])
    

models = [
    {"name": "naive", "model": MultinomialNB()},
    {"name": "logistic_regression", "model": LogisticRegression(solver="lbfgs", max_iter=KEYS["ITERATIONS"], random_state=KEYS["SEED"])},
    {"name": "svm", "model": SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=KEYS["SEED"])},
    {"name": "pac", "model":  PassiveAggressiveClassifier(max_iter=1000, random_state=KEYS["SEED"], tol=1e-3)},
]

all_scores = hp.pipeline(train[["merge", "label"]], models, basepipe, note="Base models")
#print(all_scores)
# Allegedly PAC will be the best, need ot identify what it is measuring and use it on a few ds
print("Tom Breese")

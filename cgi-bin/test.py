# Use to access the csv dataset
import pandas as pd
# Use to access the data in the csv as arrays
import numpy as np
# Need to vectorize text from the dataset as algorithm only accepts numerical input
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# Need to train and test model
from sklearn.model_selection import train_test_split
# The training models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from ml_helper.helper import Helper
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import rem_punc, tokenization, remove_stopwords, lemmatizer, make_str
import pickle
import string
string.punctuation

# Save some values in this KEYS object and refer to them leaving only one location to change
KEYS = {
    "SEED": 1,
    #"DATA_PATH": "C:\\Users\\thoma\\Documents\\test\\mscproj\\data\\fake_or_real_news.csv",
    "DATA_PATH": "C:\\Users\\thoma\\Documents\\test\\mscproj\\data\\politifact.csv",
    "TARGET": "label",
    "METRIC": "accuracy",
    "TIMESERIES": False,
    "SPLITS": 3,
    "ITERATIONS": 500,
}

#-------------------------------------
# 1. Open the dataset
#-------------------------------------
# Simplify helper object into variable
hp = Helper(KEYS)
# ds = dataset, read the CSV in the Keys
ds = pd.read_csv(KEYS["DATA_PATH"], header=0, names=["id", "title", "text", "label"])
# Merge together the two columns header and title to have a fuller text
ds["merge"] = np.array(str(ds["title"]) + str(ds["text"]))
# Create a label variable which is all of the labels in the dataset fake or real
label= np.array(ds["label"])
print(ds["label"].value_counts())

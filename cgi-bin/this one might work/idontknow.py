#!/usr/bin/env python3
# ^ Require the above to point to Python.exe - LEAVE IT IN

# Use to access the csv
import pandas as pd
# Use to access the data in the csv as arrays
import numpy as np
# spaCy for text pre-processing
import spacy as sp
# Need to vectorize text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
# Need to train and test model
from sklearn.model_selection import train_test_split
# The training model
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from ml_helper.helper import Helper
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import string
string.punctuation


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
ds = pd.read_csv('C:\\Users\\thoma\\Documents\\test\\mscproj\\data\\fake_or_real_news.csv')
#ds = pd.read_csv(KEYS["DATA_PATH"], header=0, names=["id", "title", "text", "label"])
# Merge together the two columns header and title
ds["merge"] = np.array(ds["title"] + ds["text"])
# Create a label variable which is all of the labels in the dataset fake or real
label= np.array(ds["label"])

#-------------------------------------
# PRE-PROCESS OR IMPORT DEF
#-------------------------------------

def rem_punc(text):
    # Iterate through array and return text if not included in punctuation list
    rem="".join([i for i in text if i not in string.punctuation])
    return rem
ds["new_merge"]=ds["merge"].apply(lambda x:rem_punc(x))

# ----------------------------------------------------------
# Try with and without lowering the text 
# ----------------------------------------------------------

ds["lower"]=ds['new_merge'].apply(lambda x: x.lower())


# Tokenize by splitting words into sentences
def tokenization(text):
        tokens = ''.join(text).split()
        return tokens
#applying function to the column
ds['tokenized']= ds['lower'].apply(lambda x: tokenization(x))


import nltk
#Stop words present in the library
stopwords = nltk.corpus.stopwords.words('english')

# Remove stop words from tokenized text
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output
#applying the function
ds['no_stopwords']= ds['tokenized'].apply(lambda x:remove_stopwords(x))

# Lemmatization Sequence
from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
word_lemmatizer = WordNetLemmatizer()

def lemmatizer(text):
    # Lemmatizes each word in text according to the library/corpus we have imported from nltk
    lemmed = [word_lemmatizer.lemmatize(word) for word in text]
    return lemmed

ds['lemma']=ds['no_stopwords'].apply(lambda x:lemmatizer(x))

#print(ds.head())

def make_str(list):
    str=' '.join(list)
    return str

preprocessed = []
#preprocessed=preprocessed.apply(lambda x:make_str(x))
#preprocessed.append(' '.join(make_str(row) for row in ds['lemma']))
for x in ds['lemma']:
    preprocessed.append(''.join(make_str(x)))
#print(preprocessed)

#print(merge[0])
#for x in merge:
       # proc_text=[]
        #proc_text.append(strict_tokenizer(merge[x]))


#print(proc_text[0])
#-------------------------------------
# 4. Vectorising the data
#-------------------------------------
# Count vectorizer is a built in sklearn function which converts text to a matrix of token souncts
cv = CountVectorizer()
# Learn the vocabulary dictionary and return document-term matrix, only do it to text and not label classifier
# All the text columns are now floats
x = cv.fit_transform(preprocessed)
tfidf = TfidfTransformer()
x = tfidf.fit_transform(x)


# Create a train and test ds variables, the test size is 20% and train 80%, randomise using the seed but is constant to make it reproducible
xtrain, xtest, ytrain, ytest = train_test_split(x, label, test_size=0.20, random_state=KEYS["SEED"])

# Train model using multinomial naive bayes algorithm (uses bag or words)
model = PassiveAggressiveClassifier(C = 1.0, max_iter=KEYS["ITERATIONS"], random_state=KEYS["SEED"], tol=1e-3)
# Find out what this is actually doing
model.fit(xtrain, ytrain)
# Accuracy output using the 2 testing sets - same as metric scorer?
#print(model.score(xtest, ytest))

# Used this to help: https://dataanalyticsedge.com/2019/11/26/fake-news-analysis-natural-language-processingnlp-using-python/
y_pred=model.predict(xtest)
score=accuracy_score(ytest, y_pred)
print(f'Accuracy:{round(score*100,2)}%')
print(confusion_matrix(ytest, y_pred, labels=['FAKE','REAL']))
print(f"Classification Report : \n\n{classification_report(ytest, y_pred)}")

# -------------------------------------------------------------------------------------
# With Kaggle dataset and no lemmatization we are predicting with a 95.11% accuracy
# -------------------------------------------------------------------------------------


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

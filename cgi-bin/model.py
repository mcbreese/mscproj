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
    #"DATA_PATH": "C:\\Users\\thoma\\Documents\\test\\mscproj\\data\\politifact.csv",
    "DATA_PATH": "C:\\Users\\thoma\\Documents\\test\\mscproj\\data\\snopes.csv",
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
print(ds["label"].value_counts())
# Merge together the two columns header and title to have a fuller text
ds["merge"] = np.array(ds["title"]+ ds["text"])
# Create a label variable which is all of the labels in the dataset fake or real
label= np.array(ds["label"])

#-------------------------------------
# 2. Pre-processing
#-------------------------------------

# First remove punctuation from dataset, imported from external module
ds["new_merge"]=ds["merge"].apply(lambda x:rem_punc(str(x)))

# ----------------------------------------------------------
# Try with and without lowering the text 
# ----------------------------------------------------------

# Lower the text in the dataset
#ds["lower"]=ds['new_merge'].apply(lambda x: x.lower())
# Tokenize the text so each word can be iterated through and modified if necessary
ds['tokenized']= ds['new_merge'].apply(lambda x: tokenization(x))
# Remove the stopwords
ds['no_stopwords']= ds['tokenized'].apply(lambda x:remove_stopwords(x))
# Lemmatize the text which is the process of grouping similar words together as a single form, building, builds build = build
ds['lemma']=ds['no_stopwords'].apply(lambda x:lemmatizer(x))

# Initialise a list which will be the output for each of our rows in the dataset
preprocessed = []
# Now modify the dataset (currently each row is split into lists) back to strings
for x in ds['lemma']:
    preprocessed.append(''.join(make_str(x)))

#-------------------------------------
# 3. Vectorising the data
#-------------------------------------
# Count vectorizer is a built in sklearn function which converts text to a matrix of token counts depending on the input rules
# If a word appears twice the vector for it will be 2
# Opt not to remove stop words here so we can do it centrally for dataset and test webpages
cv = CountVectorizer(lowercase=False)

# Fit_transform returns the matrix of the count vectorizer to the dataset, we need this to train later
x=cv.fit_transform(preprocessed)
# Exporting the vectorizer so it can be used externally, we don't want to transform though as thats the matrix and won't work on our input data
export_vect=cv.fit(preprocessed)
# Saved as pickle file so it can be imported
print ("Vectorizer trained. Saving vectorizer to cv.pickle")
with open("cv.pickle", "wb") as file:
    pickle.dump(export_vect, file)
print('Vectorizer saved')

# Now we want to modify the count matrix using tf-idf
# Using tf-idf instead of the count matrix of the tokes is to scale down the impact of tokens that occur frequently and are less informative than features that occur in a small fraction of the training set
tfidf = TfidfTransformer()
# Need to export as before with CV
export_tfidf = tfidf.fit(x)
x = tfidf.fit_transform(x)
print ("TFIDF trained. Saving vectorizer tfidf.pickle")
with open("tfidf.pickle", "wb") as file:
    pickle.dump(export_tfidf, file)
print('TFIDF saved')

#-----------------------------------------------
# 4. Split dataset to test and train the model
#-----------------------------------------------

# Create a train and test ds variables, the test size is 20% and train 80%, randomise using the seed but is constant to make it reproducible
xtrain, xtest, ytrain, ytest = train_test_split(x, label, test_size=0.20, random_state=KEYS["SEED"])

# Train model using Passive Aggressive Classifier which will sort the inputs into Fake or Real
#model = MultinomialNB()
model = PassiveAggressiveClassifier(C = 1.0, max_iter=KEYS["ITERATIONS"], random_state=KEYS["SEED"], tol=1e-3)
# Fit the training data to our PAC modl
model.fit(xtrain, ytrain)

# Used this to help: https://dataanalyticsedge.com/2019/11/26/fake-news-analysis-natural-language-processingnlp-using-python/
# Use predicition function on the xtest set (it sorts into the classifications)
y_pred=model.predict(xtest)
# Give an accuracy score based on the other testing set and the predictions the model provided
score=accuracy_score(ytest, y_pred)
# Print output of the accuracy of the model based on how well it guess against the tests
print(f'Accuracy:{round(score*100,2)}%')
# The confusion matrix demonstrates the False positives and False negatives to see the degree of error
print(confusion_matrix(ytest, y_pred, labels=['FAKE','REAL']))
# Print the precision, recall and f-score
print(f"Classification Report : \n\n{classification_report(ytest, y_pred)}")

# ---------------------------------------------------------------------------------------------------------------------
# 5. Output the model which can be used in the main interface, this will be quicker than training the model every time
# ---------------------------------------------------------------------------------------------------------------------

print ("Model trained. Saving model to model.pickle")
with open("model.pickle", "wb") as file:
    pickle.dump(model, file)
print('Model saved')
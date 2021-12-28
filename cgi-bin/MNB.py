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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocess import rem_punc, tokenization, remove_stopwords, lemmatizer, make_str
import string
string.punctuation

#-------------------------------------
# 1. Open the dataset
#-------------------------------------

# ds = dataset, read the CSV
ds = pd.read_csv( "C:\\Users\\thoma\\Documents\\test\\mscproj\\data\\data.csv", header=0, names=["id", "title", "text", "label"])
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
ds["lower"]=ds['new_merge'].apply(lambda x: x.lower())
# Tokenize the text so each word can be iterated through and modified if necessary
ds['tokenized']= ds['lower'].apply(lambda x: tokenization(x))
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

# Now we want to modify the count matrix using tf-idf
# Using tf-idf instead of the count matrix of the tokes is to scale down the impact of tokens that occur frequently and are less informative than features that occur in a small fraction of the training set
tfidf = TfidfTransformer()
x = tfidf.fit_transform(x)

#-----------------------------------------------
# 4. Split dataset to test and train the model
#-----------------------------------------------

# Create a train and test ds variables, the test size is 20% and train 80%, randomise using the seed but is constant to make it reproducible
xtrain, xtest, ytrain, ytest = train_test_split(x, label, test_size=0.20, random_state=1)

# Train model using MultiNomial Naive Bayes which will sort the inputs into Fake or Real
model = MultinomialNB()
# Fit the training data to our MNB model
model.fit(xtrain, ytrain)

# Use predicition function on the xtest set (it sorts into the classifications)
y_pred=model.predict(xtest)
# Give an accuracy score based on the other testing set and the predictions the model provided
score=accuracy_score(ytest, y_pred)

# Print the outputs to a file
f = open("mnb_output_lower.txt", "a")
# Print output of the accuracy of the model based on how well it guess against the tests
print(f'Accuracy:{round(score*100,2)}%', file=f)
# The confusion matrix demonstrates the False positives and False negatives to see the degree of error
print(confusion_matrix(ytest, y_pred, labels=['FAKE','REAL']), file=f)
# Print the precision, recall and f-score
print(f"Classification Report : \n\n{classification_report(ytest, y_pred)}", file=f)
f.close()
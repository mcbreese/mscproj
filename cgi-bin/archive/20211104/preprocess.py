
# Use to access the csv
#import pandas as pd
# Use to access the data in the csv as arrays
#import numpy as np
# spaCy for text pre-processing
##from ml_helper.helper import Helper
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import classification_report, confusion_matrix
# Lemmatization Sequence
from nltk.stem import WordNetLemmatizer
# String library contains a pre-defined list of punctuation
import string
string.punctuation

# ds = dataset, read the CSV in the Keys
#ds = pd.read_csv('C:\\Users\\thoma\\Documents\\test\\mscproj\\data\\test.csv')
#pd.set_option('display.max_colwidth', None)
#print(ds.head())
#print(ds["label"].value_counts())
#ds = pd.read_csv(KEYS["DATA_PATH"], header=0, names=["id", "title", "text", "label"])to
# Merge together the two columns header and title
#ds["merge"] = np.array(ds["title"])
# Create a label variable which is all of the labels in the dataset fake or real
#label= np.array(ds["label"])

#-------------------------------------
# PRE-PROCESS OR IMPORT DEF
#-------------------------------------

def rem_punc(text):
    # Iterate through array and return text if not included in punctuation list
    rem="".join([i for i in text if i not in string.punctuation])
    return rem
#ds["new_merge"]=ds["merge"].apply(lambda x:rem_punc(x))

# ----------------------------------------------------------
# Try with and without lowering the text 
# ----------------------------------------------------------

#ds["lower"]=ds['new_merge'].apply(lambda x: x.lower())


# Tokenize by splitting words into sentences
def tokenization(text):
        tokens = ''.join(text).split()
        return tokens
#applying function to the column
#ds['tokenized']= ds['lower'].apply(lambda x: tokenization(x))


import nltk
#Stop words present in the library
stopwords = nltk.corpus.stopwords.words('english')

# Remove stop words from tokenized text
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output
#applying the function
#ds['no_stopwords']= ds['tokenized'].apply(lambda x:remove_stopwords(x))


#nltk.download('wordnet')
word_lemmatizer = WordNetLemmatizer()

def lemmatizer(text):
    # Lemmatizes each word in text according to the library/corpus we have imported from nltk
    lemmed = [word_lemmatizer.lemmatize(word) for word in text]
    return lemmed

#ds['lemma']=ds['no_stopwords'].apply(lambda x:lemmatizer(x))

#print(ds.head())

def make_str(list):
    str=' '.join(list)
    return str

#preprocessed = []
#preprocessed=preprocessed.apply(lambda x:make_str(x))
#preprocessed.append(' '.join(make_str(row) for row in ds['lemma']))
#for x in ds['lemma']:
    #print(x)
    #preprocessed.append(''.join(make_str(x)))
#print(preprocessed)

#cv = CountVectorizer()
#x = cv.fit_transform(preprocessed)
#tfidf = TfidfTransformer()
#x = tfidf.fit_transform(x)
#print(len(preprocessed))
#print(len(label))
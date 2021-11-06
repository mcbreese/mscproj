# nltk required for text pre-processing
import nltk
from nltk.stem import WordNetLemmatizer
# String library contains a pre-defined list of punctuation
import string
string.punctuation

# Remove Punctuation - Iterate through array and return text if not included in punctuation list
def rem_punc(text):
    rem="".join([i for i in text if i not in string.punctuation])
    return rem

# Tokenize by splitting sentences into words
def tokenization(text):
        tokens = ''.join(text).split()
        return tokens

#Stop words present in the nltk library
stopwords = nltk.corpus.stopwords.words('english')
# Remove stop words from tokenized text
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output

# Only needed to download wordnet once, WordNetLemmatizer object links to corpus of words so it can modify the words we put in e.g. better becomes good
#nltk.download('wordnet')
word_lemmatizer = WordNetLemmatizer()
def lemmatizer(text):
    # Lemmatizes each word in text according to the library/corpus we have imported from nltk
    lemmed = [word_lemmatizer.lemmatize(word) for word in text]
    return lemmed

# Make the array thats has been pre-processed into a string once again to be processed by the model
def make_str(list):
    str=' '.join(list)
    return str
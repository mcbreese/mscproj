
# SpaCy required for pre-processing
import spacy as sp
# Require counter to modify our list of words
from collections import Counter
from string import punctuation

# Load large English corpus to recognise more words
nlp = sp.load("en_core_web_lg")

# Definition to extract the top words
def top_words(text):
    # Initialise the word variable where ALL words from article which are not stop words and included in the pos_tag variable
    words = []
    # Initialise the final list which will be our output to return to the web plugin
    topten=[]
    # NLP Corpus tags words e.g. adjective or pronoun, only want the following
    tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
    # Analyse text using nlp function and lower the input, nlp identifies what type of word it is e.g. pos (adj etc)
    doc = nlp(text.lower()) 
    for word in doc: 
        # If a stop word or the token is punctiation then continue/do nothing
        if(word.text in nlp.Defaults.stop_words or word.text in punctuation):
            continue 
        # Otherwise append to the array we previously initalised
        if(word.pos_ in tag):
            words.append(word.text)
    # Create a Counter object, this will convert the list into a dict of words and the # of occurences
    words = Counter(words)
    # Most common returns the top n from our Counter
    for word, count in words.most_common(10):
        # print (word, count)
        topten.append([word,count])
    return topten

# Scrapes page text
from test import return_text
# from preprocess import rem_punc, tokenization, remove_stopwords, lemmatizer, make_str
import spacy as sp
from collections import Counter
from string import punctuation

# Example URL just to test this page, need to import from main.py
url='https://heatworld.com/celebrity/news/molly-mae-apartment-tour/'
# Load large English corpus to recognise more words
nlp = sp.load("en_core_web_lg")

# Definition to extract the top words
def top_words(text):
    # Initialise the word variable where ALL words from article which are not stop words and included in the pos_tag variable
    words = []
    # Initialise the final list which will be our output to return to the web plugin
    topten=[]
    # NLP Corpus tags words e.g. adjective or pronoun, only want the following
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
    # Analyse text using nlp function and lower the input
    doc = nlp(text.lower()) 
    for token in doc: 
        # If a stop word or the token is punctiation then continue/do nothing
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
            continue 
        # Otherwise append to the array we previously initalised
        if(token.pos_ in pos_tag):
            words.append(token.text)
    # Create a Counter object, this will convert the list into a dict of words and the # of occurences
    words = Counter(words)
    # Most common returns the top n from our Counter
    for word, count in words.most_common(10):
        # print (word, count)
        topten.append([word,count])
    return topten

# This will be the scraped text below
# page_text=return_text(url)
# page_text=top_words(page_text)

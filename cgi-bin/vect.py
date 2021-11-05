#!/usr/bin/env python3
# ^ Require the above to point to Python.exe - LEAVE IT IN

from test import return_text
import pickle


url ='https://www.infowars.com/posts/marketing-executive-fired-for-being-white-wins-10-million-discrimination-suit/'

with open('C:\\Users\\thoma\\Documents\\test\\mscproj\\cgi-bin\\cv.pickle', "rb") as file:
    cv = pickle.load(file)

with open('C:\\Users\\thoma\\Documents\\test\\mscproj\\cgi-bin\\tfidf.pickle', "rb") as file:
    tfidf = pickle.load(file)
#
# This will be the scraped text below
#page_text=return_text(url)

def vectorize_page(text):
    data = cv.transform([text]).toarray()
    data = tfidf.transform(data)
    return data




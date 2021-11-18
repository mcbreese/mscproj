#!/usr/bin/env python3
# ^ Require the above to point to Python.exe - LEAVE IT IN

# The pickle module allows us to import and export our objects as files
import pickle
# vect module vectorises the extracted text from the web page the user is on so it can be tested against the model
from vect import vectorize_page
# Scrapes page text when given a URL
from scrape import return_text
# Topic module gets the top words to represent what the page is about - adds more detail for user
from topics import top_words
# CGI is required to execute this script rather than run it as a file
import cgi
# Numpy used in this script for sigmoid function
import numpy as np
import pandas as pd
from preprocess import rem_punc, tokenization, remove_stopwords, lemmatizer, make_str
from collections import Counter

# This print line is required otherwise it won't return properly to the JS
print("Content-Type: text/html\n\r\n")
form = cgi.FieldStorage()
# Get data from fields

url = form.getvalue('url')

# ---------------------------------------------------------------------------------------------------------
# Testing URLs
# ---------------------------------------------------------------------------------------------------------

#ds = pd.read_csv("C:\\Users\\thoma\\Documents\\test\\mscproj\\data\\fake.csv", header=0, names=["url"])
ds = pd.read_csv("C:\\Users\\thoma\\Documents\\test\\mscproj\\data\\real.csv", header=0, names=["url"])
url = ds["url"]
arr=[]

for i in url:
  # Use return text function which is saved in scrape.py
  try:
    page_text=return_text(i)
  except:
    # Error code #1, no string returned from text scraping
    print(1)
    # Quit the program


  # Next segment runs through the text pre-processing steps saved in preprocess.py, more lines of code running individually but makes it easier to understand 
  try:
    page_text=rem_punc(page_text)
    page_text.lower()
    page_text=tokenization(page_text)
    page_text=remove_stopwords(page_text)
    page_text=lemmatizer(page_text)
    page_text=make_str(page_text)
  except:
    # Error code 5, issues with text preprocessing
    print(5)

  # This function vectorises the pre-processed text from the webpage the user is on and converts it to a TF-IDF output, explained further in model.py
  data = vectorize_page(page_text)


  # Open the trained misinformation classifier model and store in a variable
  try:
    with open('C:\\Users\\thoma\\Documents\\test\\mscproj\\cgi-bin\\model.pickle', "rb") as file:
        model = pickle.load(file)
  except:
    # If the model can't be loaded then print Error Code #2
      print(2)
      
  # Define a sigmoid function as the decision_function from sklearn returns values lower and greater than 0, this converts to a confidence score in % relative to the hyperplane
  def sigmoid(x):
    return 1/(1+np.exp(-x))

  # Pass through our data from the web page to our model to receive what it's classification is
  try:  
    output=model.predict(data)
    output = np.array_str(output)
  except:
    # Error code 3, no model prediction
    print(3)
    
  # The decision function returns the confidence from the ML model of it's output
  try:
    conf_score=model.decision_function(data)
    # Convert to %
    conf_score=sigmoid(conf_score)
  except:
    # Error code 6, confidence score not produced
    print(6)
    

  arr.append(output)

count=Counter(arr)
print(arr)
print(count)



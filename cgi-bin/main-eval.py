# The pickle module allows us to import and export our objects as files
import pickle
# vect module vectorises the extracted text from the web page the user is on so it can be tested against the model
from vect import vectorize_page
# Numpy used in this script for sigmoid function
import numpy as np
from preprocess import rem_punc, tokenization, remove_stopwords, lemmatizer, make_str
from collections import Counter
import os



def open_txt(input):
    with open(input) as inp:
        data = list(inp) # or set(inp) if you really need a set
        data="".join(str(data) for data in data)
        evaluate(data, input)
    

def evaluate(data, input):
  page_text=data
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

  # Pass through our data from the web page to our model to receive what it's classification is
  try:  
    output=model.predict(data)
    output = np.array_str(output)
  except:
    # Error code 3, no model prediction
    print(3)
  
  #print(input+" "+output)
  arr.append(output)

# ==============================================================
# Evaluation of fake and real news sources
# ==============================================================
arr=[]
path= "C:\\Users\\thoma\\Documents\\test\\mscproj\\data\\fakeNewsDatasets\\fakeNewsDataset\\legit\\sample"
ext='.txt'
# Set Fake Path
for files in os.listdir(path):
    if files.endswith(ext):
        open_txt(path+"\\"+files)
        
    else:
        continue

count=Counter(arr)

f = open("eval_output.txt", "a")
print("Total classifications from a REAL dataset:", file=f)
print(count, file=f)


# Set Real Path
arr=[]
path= "C:\\Users\\thoma\\Documents\\test\\mscproj\\data\\fakeNewsDatasets\\fakeNewsDataset\\fake\\sample"
ext='.txt'
# Set Fake Path
for files in os.listdir(path):
    if files.endswith(ext):
        open_txt(path+"\\"+files)
        
    else:
        continue

count=Counter(arr)
print("Total classifications from a FAKE dataset:", file=f)
print(count, file=f)
f.close()



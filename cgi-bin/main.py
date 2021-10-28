#!/usr/bin/env python3
# ^ Require the above to point to Python.exe - LEAVE IT IN

import pickle
# --------------------------------
# Below breaks script bc of opening csv file script is run on import
# --------------------------------
from idontknow import vectorize_page
# Scrapes page text
from test import return_text
import cgi
import numpy as np

print("Content-Type: text/html\n\r\n")
form = cgi.FieldStorage()
# Get data from fields
url = form.getvalue('url')
#Real
#url='https://www.theguardian.com/environment/2021/oct/28/world-failing-make-changes-avoid-climate-breakdown-report'
#Fake 
#url='https://www.theburningplatform.com/2021/10/11/the-vaccine-mandate-is-a-hoax/'
url='https://heatworld.com/celebrity/news/molly-mae-apartment-tour/'
#url='https://solarsystem.nasa.gov/solar-system/sun/overview/'
#url ='https://www.infowars.com/posts/marketing-executive-fired-for-being-white-wins-10-million-discrimination-suit/'
print('Python URL below')
print(url)


# This will be the scraped text below
page_text=return_text(url)
data = vectorize_page(page_text)
#print(data)
#Open our saved model in the .pickle file
# --------------------------------
# Below just needed explicit file path in order to work
# --------------------------------
with open('C:\\Users\\thoma\\Documents\\test\\cgi-bin\\idk.pickle', "rb") as file:
    model = pickle.load(file)
    
# Define a sigmoid function as the decision_function from sklearn returns values lower and greater than 0, this converts to a %
def sigmoid(x):
  return 1/(1+np.exp(-x))


# Pass through our data from the web page to our model to receive what it's classification is
output=model.predict(data)
# The decision function returns the confidence from the ML model of it's output - anything above 0 is still a yes
conf_score=model.decision_function(data)
conf_score=sigmoid(conf_score)
print(f'The model has returned {output} with a confidence of {conf_score}')
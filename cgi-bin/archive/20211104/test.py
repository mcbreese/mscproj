#!/usr/bin/env python3
# ^ Require the above to point to Python.exe - LEAVE IT IN



#import cgi
import requests
from bs4 import BeautifulSoup

#form = cgi.FieldStorage()

# Get data from fields
#url = form.getvalue('url')



# Scrape the page text
# Use the requests library to access the webpage content
# Use beautiful soup to identify the textual data

# Use requests library to request the webpage content

def return_text(url):
        # Open the page content
	html=requests.get(url).content
	# Scrape using lxml tree library: https://stackabuse.com/introduction-to-the-python-lxml-library/
	scrape=BeautifulSoup(html, 'lxml')
	# List of textual elements obtained from here: https://flaviocopes.com/html-text-tags/
	text_elements=['p', 'strong', 'em', 'b', 'u', 'i', 'h1', 'h2', 'h3', 'h4','h5','h6','span','q','li']
	text_out=""

        # Search through the page, if the HTML tage is listed in our above array then add it to the text_out variable
	for i in scrape.find_all(text=True):
		if i.parent.name in text_elements:
			text_out+= '{} '.format(i)
	
	escape_sym=['\r','\n','\t','\xa0']

        # Replace any of the above values in our text_out array with blank, they are escape characters
	for e in escape_sym:
		text_out=text_out.replace(e, '')
	
	return text_out



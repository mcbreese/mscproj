# Requests library required to request the webpage content
import requests
# Beautiful soup scrapes the text
from bs4 import BeautifulSoup

def return_text(url):
    # Open the page content
	html=requests.get(url).content
	scrape=BeautifulSoup(html, 'lxml')
	# List of textual elements obtained from here: https://flaviocopes.com/html-text-tags/
	#text_elements=['p', 'strong', 'em', 'b', 'u', 'i', 'h1', 'h2', 'h3', 'h4','h5','h6','span','q','li']
	text_elements=['p', 'strong', 'em', 'b', 'u', 'i', 'h1', 'h2', 'h3']
	text_out=""
    # Search through the page, if the HTML tage is listed in our above array then add it to the text_out variable
	for elem in scrape.find_all(text=True):
		if elem.parent.name in text_elements:
			text_out+= '{} '.format(elem)
	
	escape_sym=['\r','\n','\t','\xa0']
    # Replace any of the above values in our text_out array with blank, they are escape characters
	for e in escape_sym:
		text_out=text_out.replace(e, '')
	
	return text_out



# Need to import the pickle module to open the objects saved as files
import pickle

# The Count Vectorizer transformer as a file
with open('C:\\Users\\thoma\\Documents\\test\\mscproj\\cgi-bin\\cv.pickle', "rb") as file:
    cv = pickle.load(file)

# The TF-IDF transformer as a file
with open('C:\\Users\\thoma\\Documents\\test\\mscproj\\cgi-bin\\tfidf.pickle', "rb") as file:
    tfidf = pickle.load(file)

# The processed page text from the website is passed into this function
def vectorize_page(text):
    # We require the count vectorizer transformer that has been trained with our training data or the results will be random since vocabularies are different 
    data = cv.transform([text]).toarray()
    # The same applies to the tf-idf transformer
    data = tfidf.transform(data)
    return data
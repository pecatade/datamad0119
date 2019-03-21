
##### TRUE AND FALSE NEWS DETERTOR ######

# The app get a new's url and detect if the new is true or flase
#       



if __name__ == '__main__':


  # Libraries importation.

  import requests
  from bs4 import BeautifulSoup
  import pandas as pd
  import urllib.request
  from urllib.request import urlopen
  import re
  import json
  import numpy as np
  import urllib.request
  import urllib.parse
  import urllib.error
  import pickle
  from sklearn.feature_extraction.text import CountVectorizer
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.naive_bayes import MultinomialNB
  import sklearn.metrics 
  import matplotlib.pyplot as plt
  from sklearn.model_selection import train_test_split
  from sklearn.linear_model import PassiveAggressiveClassifier
  from sklearn.linear_model import LogisticRegression
  from sklearn.svm import LinearSVC
  from sklearn.feature_extraction.text import HashingVectorizer
  from sklearn.metrics import accuracy_score
  import numpy as np
  import itertools
  import pandas as pd
  import re
  import string
  import nltk
  from nltk import sent_tokenize, word_tokenize
  from nltk.stem import WordNetLemmatizer
  from nltk.stem import *
  from nltk.corpus import stopwords
  from nltk.probability import FreqDist
  from nltk.corpus import names
  import random
  from functions_transf import clean_up_text
  from functions_transf import news_scraper
  

  # Asking for a new to analyze

  url = str(input('Enter an url: '))

  # load the vectorizer from disk
  filename = '../finalized_vectorizer.sav'
  loaded_vectorizer = pickle.load(open(filename, 'rb'))

  # load the model from disk
  filename = '../finalized_model.sav'
  loaded_model = pickle.load(open(filename, 'rb'))

  # Scraping the new 

  new_scraped = news_scraper(url)

  # Cleaning the text

  to_predict = [clean_up_text(new_scraped['body'])]

  # Loading the vectorizer

  X = loaded_vectorizer.transform(to_predict)

  # Loading the classifier and making the prediction
  
  new_label = loaded_model.predict(X)

  # Printing the label of the new

  print('\x1b[6;30;42m' + str(new_label) + '\x1b[0m')
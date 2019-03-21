# libraries importation
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

# Building a function for scraping the targeted news.

def news_scraper(url):
    '''
    Input: a string with the URL of the new
    Output: a dictionary with thes key-value pairs:
        'url': a string with the new's url 
        'h1': a string with the headline of the new
        'h2': a string with the subtitle of the new
        'author': a string with the author of the new
        'body': a string with the body of the new
    '''
    
    new_scraped = {}
    html = requests.get(url).content
    soup = BeautifulSoup(html, "lxml")
    if 'elpais.com' in url:
        new_scraped['url'] = url
        new_scraped['h1'] = ''.join(element.text for element in soup.find_all('h1',{'itemprop':"headline"}))
        new_scraped['h2'] = ''.join(element.text for element in soup.find_all('h2',{'itemprop':"alternativeHeadline"}))
        new_scraped['author'] = ''.join(element.text.replace("\n","") for element in soup.find_all('span',{'class':"autor-nombre"}))
        new_scraped['body'] = ''.join(element.text.replace('\n', '') for element in soup.find_all('p')).split('NEWSLETTER')[0]
        new_scraped['label'] = True
        
    elif 'elmundo.es' in url:
        new_scraped['url'] = url
        new_scraped['h1'] = ''.join(element.text for element in soup.find_all('h1'))
        new_scraped['h2'] = ''.join(element.text for element in soup.find_all('p',{'class':"ue-c-article__standfirst"}))
        new_scraped['author'] = ''.join(element.text.replace("\n","") for element in soup.find_all('div',{'class':"ue-c-article__byline-name"}))
        new_scraped['body'] = ''.join(element.text.replace('\n', ' ') for element in soup.find_all('div',{'class':"ue-l-article__body ue-c-article__body"})).split('Conforme a los criterios deThe Trust Project')[0]
        new_scraped['label'] = True
    
    elif 'abc.es' in url:
        new_scraped['url'] = url
        new_scraped['h1'] = ''.join(element.text for element in soup.find_all('span',{'class':"titular"}))
        new_scraped['h2'] = ''.join(element.text for element in soup.find_all('h2',{'class':"subtitulo"}))
        new_scraped['author'] = ''.join(element.text for element in soup.find_all('a',{'class':"autor"}))
        new_scraped['body'] = ''.join(element.text.replace("''", '') for element in soup.find_all('p')).split('¡Hola, !')[0]
        new_scraped['label'] = True
        
    elif 'lavozdegalicia.es' in url:
        new_scraped['url'] = url
        new_scraped['h1'] = ''.join(element.text for element in soup.find_all('h1',{'itemprop':"headline"}))
        new_scraped['h2'] = ''.join(element.text for element in soup.find_all('h2',{'itemprop':"alternativeHeadline"}))
        new_scraped['author'] = ''.join(element.text.replace('\n','') for element in soup.find_all('span',{'class':"author"})).replace('\t', '')
        new_scraped['body'] = ''.join(element.text for element in soup.find_all('p')).split('Hemos creado para ti una selección de contenidos para que los recibas')[0]
        new_scraped['label'] = True
    
    elif 'lavanguardia.com' in url:
        new_scraped['url'] = url
        new_scraped['h1'] = ''.join(element.text for element in soup.find_all('h1',{'itemprop':"headline"}))
        new_scraped['h2'] = ''.join(element.text for element in soup.find_all('h2',{'itemprop':"alternativeHeadline"}))
        new_scraped['author'] = ''.join(element.text for element in soup.find_all('span',{'itemprop':"name"}))
        new_scraped['body'] = ''.join(element.text.replace('\n', '') for element in soup.find_all('div',{'itemprop':"articleBody"}))
        new_scraped['label'] = True
  
    elif 'elmundotoday.com' in url:
        new_scraped['url'] = url
        new_scraped['h1'] = ''.join(element.text for element in soup.find_all('h1',{'class':"entry-title"}))
        new_scraped['h2'] = ''.join(element.text for element in soup.find_all('p',{'class':"td-post-sub-title"}))
        new_scraped['author'] = ''.join(element.text.replace('Por', '') for element in soup.find_all('div',{'class':"td-post-author-name"}))
        new_scraped['body'] = ''.join(element.text.replace('\n', '') for element in soup.find_all('div',{'class':"td-pb-span10"}))
        new_scraped['label'] = False
    
    elif 'alertadigital.com' in url:
        new_scraped['url'] = url
        new_scraped['h1'] = ''.join(element.text for element in soup.find_all('h2'))
        new_scraped['h2'] = None
        new_scraped['author'] = ''.join(element.text for element in soup.find_all('div',{'id':"datemeta_r"})).split('|')[0]
        new_scraped['body'] = ''.join(element.text.replace('\n', '') for element in soup.find_all('div',{'class':"entry"})).replace('\xa0', '')
        new_scraped['label'] = False
    
    elif 'okdiario.com' in url:
        new_scraped['url'] = url
        new_scraped['h1'] = ''.join(element.text for element in soup.find_all('h1',{'class':"entry-title"}))
        new_scraped['h2'] = ''.join(element.text for element in soup.find_all('h2',{'itemprop':"alternativeHeadline"}))
        new_scraped['author'] = ''.join(element.text.replace('\n', '') for element in soup.find_all('li',{'class':"author-name"}))
        new_scraped['body'] = ''.join(element.text.replace('\n', '') for element in soup.find_all('div',{'class':"entry-content"}))
        new_scraped['label'] = False
    
    elif 'mediterraneodigital.com' in url:
        new_scraped['url'] = url
        new_scraped['h1'] = ''.join(element.text.replace('\t', '') for element in soup.find_all('h2',{'class':"contentheading"})).replace('\n', '')
        new_scraped['h2'] = None
        new_scraped['author'] = ''.join(element.text.replace('\t', '') for element in soup.find_all('dd',{'class':"createdby"})).replace('Escrito por', '').replace('\n', '')
        new_scraped['body'] = (''.join(element.text for element in soup.find_all('p',{'style':"text-align: justify;"}))).split('\xa0©\xa0GRUP')[0]
        new_scraped['label'] = False
    
    elif 'periodistadigital.com' in url:
        new_scraped['url'] = url
        new_scraped['h1'] = ''.join(element.text for element in soup.find_all('h1'))
        new_scraped['h2'] = ''.join(element.text for element in soup.find_all('p',{'class':"subtitle"}))
        new_scraped['author'] = (''.join(element.text for element in soup.find_all('div',{'class':"page-header-author text-left"}))).split(',')[0].replace('\n', '')
        new_scraped['body'] = ''.join(element.text for element in soup.find_all('div',{'class':"text-block"})).replace('\n', '')
        new_scraped['label'] = False
  
    elif 'haynoticia.es' in url:
        new_scraped['url'] = url
        new_scraped['h1'] = ''.join(element.text.replace('\n', '') for element in soup.find_all('h1',{'class':"entry-title"})).replace('\t', '')
        new_scraped['h2'] = None
        new_scraped['author'] = ''.join(element.text.replace("\n","") for element in soup.find_all('span',{'class':"author vcard"}))
        new_scraped['body'] = ''.join(element.text.replace('\xa0', ' ') for element in soup.find_all('p')).replace('No creas todo lo que lees por Internet', '')
        new_scraped['label']  = False
    
    elif 'thepatriota.com' in url:
        new_scraped['url'] = url
        new_scraped['h1'] = ''.join(element.text for element in soup.find_all('h1',{'class':"entry-title"}))
        new_scraped['h2'] = None
        new_scraped['author'] = ''.join(element.text.replace('Por','') for element in soup.find_all('div',{'class':"td-post-author-name"})).replace('-', '')
        new_scraped['body'] = ''.join(element.text for element in soup.find_all('p')).split('Nuestra web es un sitio de humor')[0]
        new_scraped['label'] = False
              
    
    return new_scraped


# function for cleaning

def clean_up_text(text):
    """
    The function cleans up numbers, remove punctuation and line break, and special characters from a string 
    and converts it to lowercase.

    Args:
        text: The string to be cleaned up.

    Returns:
        A string that has been cleaned up.
    """
    text = text.lower()
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\n', ' ', text) 
    text = re.sub('\w*\d\w*', '', text)    
    text = re.sub('[‘’“”…«»¿?¡!\-_\(\)]', '', text)
    text = re.sub(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', '', text)
  
    return text


# function for tokenizing

def tokenize_text(text):
        """
        Tokenize a string.

        Args:
            text: String to be tokenized.

        Returns:
            A list of words as the result of tokenization.
        """
        return word_tokenize(text)

# function for stemming, and lemmatizing

def stem_and_lemmatize(list_of_words):
    """
    Perform stemming and lemmatization on a list of words.

    Args:
        list_of_words: A list of strings.

    Returns:
        A list of strings after being stemmed and lemmatized.
    """
    stemmer = nltk.stem.SnowballStemmer('spanish')
    lemmatizer = WordNetLemmatizer()
    stemmed_lemmantized_list = [stemmer.stem(lemmatizer.lemmatize(word)) for word in list_of_words]
    return stemmed_lemmantized_list

    # function for stops words

def remove_stopwords(list_of_words):
    """
    Remove English stopwords from a list of strings.

    Args:
        list_of_words: A list of strings.

    Returns:
        A list of strings after stop words are removed.
    """
    spanish_stop_words = stopwords.words('spanish')
      
    return [w for w in list_of_words if not w in spanish_stop_words]
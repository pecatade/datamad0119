import speech_recognition as sr
import subprocess
import os
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from word2number import w2n
from unicodedata import normalize
import re
import funciones as fu

# Código principal
variable = True
#name = fu.name_set()
#fu.say("Hola {}, que puedo hacer por ti?. Di mi nombre y lo que quieres que haga para que me active.".format(name))

while variable:
    try:
        transcript = fu.activate()
        if transcript[0] == True:
            print("T: " + transcript[1])
            try:
                if "pesada" in transcript[1]:
                    variable = False
                else:
                    if fu.pd_fun(transcript[1]) == fu.load_csv:
                        df = fu.load_csv()
                    else:
                        fu.pd_fun(transcript[1])(transcript[1], df)
                    
            except sr.UnknownValueError:
                fu.say("Vocaliza un poquito, que no hay quien te entienda")
            except sr.RequestError as e:
                print("Could not request results from Mandy service; {0}".format(e))
            
        else:
            continue
    
    except:
        continue
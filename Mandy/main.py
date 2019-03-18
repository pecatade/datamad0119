import speech_recognition as sr
import subprocess
import os
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

r = sr.Recognizer()
r.energy_threshold = 2500
mic = sr.Microphone()

# Funciones para recoger el sonido y transformarlo:
def say(text):
    subprocess.call(['say', "-r 180", text])

# obtain audio from the microphone and transcript
def fun_transcript(t=5, p=6, l="es-ES"):
    try:
        with mic as source:
            audio = r.listen(source, timeout=t, phrase_time_limit=p)
            transcript = r.recognize_google(audio, language = l)
            return transcript.lower()
    except:
        pass
    
# activate when you call mandy and return
def activate(phrase='mandy'):
    try:
        transcript = fun_transcript(t=5, p=6)
        if phrase in transcript.lower():
            return [True, transcript.lower()]
        else:
            return activate()
    except:
        return [False, ""]

# personaliza el nombre de la persona con la que habla
def name_set():
    try:
        say("Hola colega, no nos conocemos, ¿cual es tu nombre?")
        n = fun_transcript(t=3, p=3)
        if n == None:
            say("Estoy sorda, ¿puedes repetirlo?")
            n = fun_transcript(t=4, p=3)
            return n
        else:
            return n
    except:
        pass

# Funciones de limpieza de la frase
def tokenize(s):
    return word_tokenize(s)

def stem_and_lemmatize(s):
    return [WordNetLemmatizer().lemmatize(SnowballStemmer("spanish").stem(x)) for x in tokenize(s)]

def remove_stopwords(s):
    return " ".join([w for w in stem_and_lemmatize(s) if not w in stopwords.words("spanish")])

# Llama a la función correspondiente en función de tus palabras:
def pd_fun(trans):
    dicfun = { 
        1: [dftypes, ["tip","tod","column"]], 
        2: [dfcolumntype,["tip","column"]],
        3: [shape,["dimension"]]
             }
    
    for x in dicfun:
        if all(w in trans for w in dicfun[x][1]):
            return dicfun[x][0]()

# Lista de funciones de pandas
# Show you the list of possible csv and you choice one
def load_csv():
    print(os.listdir("./datasets"))
    say("Los dataset disponibles son: " + " ".join(os.listdir("./datasets")) + " Elige el que quieras que abra")
    try:
        transcript = fun_transcript(t=6, p=6)
        while transcript == None:
            say("Estoy sorda, ¿puedes repetirlo?")
            transcript = fun_transcript(t=5, p=6)
        df = pd.read_csv("./datasets/{}.csv".format(transcript))
        print(transcript)
        display(df.head())
        return df
    except:
        pass

def head():
    try:
        display(df.head())
    except:
        pass

def tail():
    try:
        display(df.tail())
    except:
        pass
    
def shape():
    try:
        print(df.shape)
        say("Este dataset tiene {} columnas y {} filas".format(df.shape[1], df.shape[0]))
    except:
        pass
    
def dftypes():
    try:
        print(df.dtypes)
        say("Aquí tienes un listado de todas las columnas con sus tipos de datos".format(df.shape[1], df.shape[0]))
    except:
        pass

def dfcolumntype():
    try:
        column = trans.split(" ")[-1]
        print(df[column].dtypes)
        say("El tipo de data de la columna {} es {}".format(column, df[column].dtypes))
    except:
        pass

def columnas():
    try:
        display(df.columns)
    except:
        pass

def iloca():
    try:
        display(df.iloc[])
    except:
        pass

def loca():
    try:
        display(df.loc[])
    except:
        pass

def isnullo():
    try:
        display(df.isnull().sum())
    except:
        pass

def dropnulos():
    try:
        return df.dropna(subset=["a"], inplace=True)
    except:
        pass

def fillnulos():
    try:
        return df["a"].fillna(value=)
    except:
        pass
    
def changetype():
    try:
        return df["a"].astype(float)
    except:
        pass

def renombrar():
    try:
        return df.rename(columns={'old_name': 'new_ name'})
    except:
        pass

def renombrar():
    try:
        return df.set_index('column_one')
    except:
        pass

def descrip():
    # Summary statistics for numerical columns
    try:
        display(df.describe())
    except:
        pass

def media():
    # Returns the mean of all columns
    try:
        display(df.mean())
    except:
        pass
    
def correlacion():
    # Returns the correlation between columns in a DataFrame
    try:
        display(df.corr())
    except:
        pass
    
def counter():
    # Returns the number of non-null values in each DataFrame column
    try:
        display(df.count())
    except:
        pass
    
def maximus():
    # Returns the highest value in each column
    try:
        display(df.max())
    except:
        pass
    
def minimas():
    # Returns the lowest value in each column
    try:
        display(df.min())
    except:
        pass

def mediana():
    # Returns the median of each column
    try:
        display(df.median())
    except:
        pass

# Código principal
variable = True
name = name_set()
say("Hola {}, que puedo hacer por ti?. Di mi nombre y lo que quieres que haga para que me active".format(name))

while variable:
    try:
        transcript = activate()
        print(r.energy_threshold)
        if transcript[0] == True:
            try:
                print(transcript[1])
                if "abre" in transcript[1] and "dataset" in transcript[1]:
                    df = load_csv()
                if "calla" in transcript[1]:
                    variable = False
                clean = remove_stopwords(transcript[1])
                print(clean)
                pd_fun(clean)
            except sr.UnknownValueError:
                say("Vocaliza un poquito, que no hay quien te entienda")
            except sr.RequestError as e:
                print("Could not request results from Mandy service; {0}".format(e))
            
        else:
            pass
    except:
        pass
    
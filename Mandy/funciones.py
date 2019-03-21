import speech_recognition as sr
import subprocess
import os
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from word2number import w2n
from unicodedata import normalize
from keraslearning import traineural
import re

r = sr.Recognizer()
r.energy_threshold = 1800
mic = sr.Microphone()

# Funciones para recoger el sonido y transformarlo:
def say(text):
    subprocess.call(['say', "-r 180", text])

# obtain audio from the microphone and transcript
def fun_transcript(t=6, p=6, l="es-ES"):
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
        transcript = fun_transcript(t=6, p=6)
        if phrase in transcript.lower():
            return [True, transcript.lower()]
        else:
            return activate()
    except:
        return [False, ""]

# Llama a la función correspondiente en función de tus palabras:
def pd_fun(trans):
    trans = normalize('NFC', re.sub(r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1", normalize("NFD", trans), 0, re.I))
    dic = {
        load_csv:"load_csv",
        header:"header",
        tailer:"tailer",
        shape:"shape",
        dftypes:"dtype",
        columnas:"columnas",
        iloca:"iloca",
        loca:"loca",
        isnullo:"isnullo",
        dropnulos:"dropnulos",
        fillnulos:"fillnulos",
        changetype:"changetype",
        renombrar:"renombrar",
        newindex:"newindex",
        descrip:"descrip",
        media:"media",
        correlacion:"correlacion",
        counter:"counter",
        maximus:"maximus",
        minimas:"minimas",
        mediana:"mediana",
    }
    num = ["0","1","2","3","4","5","6","7","8","9"]
    for x in num:
        trans = trans.replace(x, "")
    print(traineural(trans))
    for x in dic:
        if dic[x] == traineural(trans):
            return x
        else:
            pass

# personaliza el nombre de la persona con la que habla
def nameset():
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
    

# Lista de funciones de pandas
# Show you the list of possible csv and you choice one
def load_csv():
    lista = [file.replace(".csv", "").replace(".json", "") for file in os.listdir("./datasets") if file.endswith(".csv") or file.endswith(".json")]
    print(lista)
    say("Los dataset disponibles son: " + " ".join(lista) + " Elige el que quieras que abra")
    transcript = fun_transcript(t=6, p=6)
    print(transcript)
    while transcript == None:
        say("Estoy sorda, ¿puedes repetirlo?")
        transcript = fun_transcript(t=6, p=6)
    df = pd.read_csv("./datasets/{}.csv".format(transcript))
    df.columns = [x.lower() for x in df.columns]
    return df

def header(trans, df):
    try:
        print(df.head())
    except:
        pass

def tailer(trans, df):
    try:
        print(df.tail())
    except:
        pass
    
def shape(trans, df):
    try:
        print(df.shape)
        say("Este dataset tiene {} columnas y {} filas".format(df.shape[1], df.shape[0]))
    except:
        pass
    
def dftypes(trans, df):
    try:
        print(df.dtypes)
        say("Aquí tienes un listado de todas las columnas con sus tipos de datos")
    except:
        pass

def columnas(trans, df):
    try:
        print(df.columns)
        say("Las columnas son: {}".format(" ".join(list(df.columns))))
    except:
        pass

def iloca(trans, df):
    try:
        n = trans.split(" ")[-1]
        print(df.iloc[int(n)])
    except:
        say("¿Cual es el indice de la fila que buscas?")
        n = fun_transcript(t=5, p=5)
        print(df.iloc[int(n)])
        pass

def loca(trans, df):
    try:
        say("¿En qué columna quieres buscar?")
        column = fun_transcript(t=10, p=5)
        say("¿Qué valor estás buscando?")
        equal = fun_transcript(t=10, p=5)
        print(df.loc[column == equal])
    except:
        say("¿En qué columna quieres buscar?")
        column = input("¿En qué columna estás buscando?")
        say("¿Qué valor estás buscando?")
        equal = input("¿Qué valor estás buscando?")
        print(df.loc[column == equal])

def isnullo(trans, df):
    try:
        print(df.isnull().sum()[df.isnull().sum() > 0])
    except:
        pass

def dropnulos(trans, df):
    try:
        say("Nulos eliminados")
        return df.dropna(inplace=True)
    except:
        pass

def fillnulos(trans, df):
    try:
        say("¿En qué columna deseas sustituir los valores nulos?")
        column = fun_transcript(t=5, p=5)
        say("¿Porque valor deseas sustituir los nulos de la columna {}?".format(column))
        values = fun_transcript(t=5, p=5)
        return df[column].fillna(value = values, inplace=True)
    except:
        say("¿En qué columna deseas sustituir los valores nulos?")
        column = input("¿En qué columna deseas sustituir los valores nulos?")
        say("¿Porque valor deseas sustituir los nulos de la columna {}?".format(column))
        values = input("¿Porque valor deseas sustituir los nulos de la columna {}?".format(column))
        return df[column].fillna(value = values, inplace=True)
    
def changetype(trans, df):
    try:
        say("¿A que columna quieres cambiar el tipo?")
        column = fun_transcript(t=5, p=5)
        say("¿A que tipo quieres cambiarlo?")
        types = fun_transcript(t=5, p=5)
        return df[column].astype(types)
    except:
        say("¿A que columna quieres cambiar el tipo?")
        column = input("¿A que columna quieres cambiar el tipo?")
        say("¿A que tipo quieres cambiarlo?")
        types = input("¿A que tipo quieres cambiarlo?")
        return df[column].astype(types)

def renombrar(trans, df):
    try:
        say("¿A que columna quieres cambiar el nombre?")
        old = fun_transcript(t=5, p=5)
        say("¿Cómo quieres que se llame la columna?")
        new = fun_transcript(t=5, p=5)
        return df.rename(columns={old: new}, inplace=True)
    except:
        say("¿A que columna quieres cambiar el nombre?")
        old = input("¿A que columna quieres cambiar el nombre?")
        say("¿Cómo quieres que se llame la columna?")
        new = input("¿Cómo quieres que se llame la columna?")
        return df.rename(columns={old: new}, inplace=True)

def newindex(trans, df):
    try:
        say("¿Qué columna quieres que sea el nuevo índice?")
        column = fun_transcript(t=5, p=5)
        return df.set_index(column, inplace=True)
    except:
        say("¿Qué columna quieres que sea el nuevo índice?")
        column = input("¿Qué columna quieres que sea el nuevo índice?")
        return df.set_index(column, inplace=True)

def descrip(trans, df):
    # Summary statistics for numerical columns
    try:
        print(df.describe())
    except:
        pass

def media(trans, df):
    # Returns the mean of all columns
    try:
        print(df.mean())
    except:
        pass
    
def correlacion(trans, df):
    # Returns the correlation between columns in a DataFrame
    try:
        print(df.corr())
    except:
        pass
    
def counter(trans, df):
    # Returns the number of non-null values in each DataFrame column
    try:
        print(df.count())
    except:
        pass
    
def maximus(trans, df):
    # Returns the highest value in each column
    try:
        print(df.max())
    except:
        pass
    
def minimas(trans, df):
    # Returns the lowest value in each column
    try:
        print(df.min())
    except:
        pass

def mediana(trans, df):
    # Returns the median of each column
    try:
        print(df.median())
    except:
        pass

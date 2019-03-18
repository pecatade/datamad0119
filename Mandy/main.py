import speech_recognition as sr
import subprocess
import os
import pandas as pd

r = sr.Recognizer()
r.energy_threshold = 2200
mic = sr.Microphone()
        
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

def pd_fun(trans):
    dicfun = {"shape": shape(), "tipos de todas las columnas": dftypes(), 
              "tipo de la columna": dfcolumntype()}
    for x in dicfun:
        if x in trans:
            return dicfun[x]

         
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
                pd_fun(transcript[1])
            except sr.UnknownValueError:
                say("Vocaliza un poquito, que no hay quien te entienda")
            except sr.RequestError as e:
                print("Could not request results from Mandy service; {0}".format(e))
            
        else:
            pass
    except:
        pass
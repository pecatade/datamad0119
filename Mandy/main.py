import funciones as fun
import speech_recognition as sr

# Código principal
variable = True
fun.say("Estoy lista. Di mi nombre y lo que quieres que haga para que me active.")
print("Estoy lista. Di mi nombre y lo que quieres que haga para que me active.")

while variable:
    try:
        transcript = fun.activate()
        if transcript[0] == True:
            print("T: " + transcript[1])
            try:
                if "pesada" in transcript[1]:
                    variable = False
                elif transcript[1] == "Mandy":
                    pass
                else:
                    if fun.pd_fun(transcript[1]) == fun.load_csv:
                        df = fun.load_csv()
                    else:
                        fun.pd_fun(transcript[1])(transcript[1], df)
                    
            except sr.UnknownValueError:
                fun.say("Vocaliza un poquito, que no hay quien te entienda")
            except sr.RequestError as e:
                print("Could not request results from Mandy service; {0}".format(e))
            
        else:
            pass
    
    except:
        pass
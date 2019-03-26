![Mandy logo](https://i.imgur.com/eu5x9zf.jpg)


# MANDY: VOICE CONTROL VIRTUAL ASSISTANT FOR PANDAS


## OBJECTIVE:

This project pretend to develop a virtual assistant who allows everyone to analyze data without programming.



## STEPS:

- The first step was to listen for user voice constantly and then transcribe the voice to text when Mandy is called.

- The second step was to build a neuronal network that understands the differents orders.
    
    - To build the neuronal network the first step was to create a dataset with training samples.
    
    - Secondly we use the Baby RNN model from Keras for training.
    
    - The last, is to create a function to make predictions in real time for the transcriptions.
 
- The third step was to create a function for each funcionality that we want Mandy to do.

- The last was to put everything together and Mandy is done 😁.



## HOW TO RUN MANDY:

Just run the main.py in the terminal from the repository.

    python3 main.py
    


## FILES IN THE REPOSITORY:

Directory with the .py neccesary files to execute the app for fake news detection:

    funciones.py
    main.py

Also you can find two folders in the directory:

    datasets
    model
    
The datasets folder is used to stock the datasets that Mandy is able to open.

In the model folder there are all the necessary files to train the neuronal network.

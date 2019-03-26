![Mandy logo](https://i.imgur.com/a/80vz1V6)


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



## FILES IN THE REPOSITORY:

Modular, directory with the .py neccesary files to execute the app for fake news detection:

	functions_transf.py
	main.py

finalized_model.sav and finalized_vectorizer.sav the file where are stored de vectorizer and the trained model  for the classifier.

main.ipynb: where is developed the code to make the machine learning model.

news_scraping.ipynb: it contains the code to build the dataset.

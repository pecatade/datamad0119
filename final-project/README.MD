![IronHack Logo](https://s3-eu-west-1.amazonaws.com/ih-materials/uploads/upload_d5c5793015fec3be28a63c4fa3dd4d55.png)

# NATURAL LANGUAGE PROCESSING FOR FAKE NEWS DETECTION: 

    **Objective**:

- This project pretend to develop a fake news detector.

    - For this project we have to build an dataset because of the           
      inexistence of such a corpus in Spanish.

    - Once we hava a dataset, we have develop a machine learning model to  
      classify the news: false or true.
    
    - So, we are in front of a supervised learning challenge.

## Competitive advantage:

-The innovation is found in the fact that the project is specifically directed to news in Spanish because. 

    - There is no other detector targeted to this language.


## Creation of the defintive dataset

We have created the **dataset** making web scraping for real and false news. 

We take the **real news** from different spanish dailies (by web scraping) whose reputation about their veracity has been demonstrated along the years. 

We find the **false news** orienting by the maldita.es web whose goal is to research about the falsity of some news. We get the ones that are marked as false by maldita.es team (a web whose porpose is to research about false online news and rumours). Some has been scraped by hand for the ineffectiveness of coding for a python scrap because of are very few news for each particular newspaper, but the most of them have been obtained by web scraping.

    **Result**
    A dataframe with the next record:

    762 rows for 762 new.
    
    Each record has the next columns:
        -Author: journalist who wrote the new.
        -Body: text of the new
        -h1: headline
        -h2. subheadline.
        -url
    Besides, we added two more columns / features:
        We build two new columns (as the result of exploratory data analysis)

        -new_structure:     value 1 (if the new has the traditional 
                            structure: headline, subheadline, 
                            body and author) or 
                    
                            value 0 (if there is an 'empty' value in one of these columns).

        -body length:        the length of the new's body in number of characters. 

# We have built a function for scraping the targeted news.

**news_scraper**(url):
    '''
    Input: a string with the URL of the new
    Output: a dictionary with thes key-value pairs:
        'url': a string with the new's url 
        'h1': a string with the headline of the new
        'h2': a string with the subtitle of the new
        'author': a string with the author of the new
        'body': a string with the body of the new
    

## CREATING MODEL FOR CLASSIFICATION

For this task, we will combine differente ways of processing text (vectorizer) with various classifiers.

**Vectorizers**:

-ad hoc tokenization, stemming and lemmatization (we have built four functions for that).

-vectorization by three different models: (TFIDF vectorizer, Countvectorizer and HashingVectorizer).

**Classifiers**:

    Nltk.NaiveBayes.
    MultinomialNB
    PassiveAggresive.
    LogisticRegression.
    LinearSVC

So, we will test (by accuracy and confusion matrix) the next mixes of vectorizer and classficator:

    Ad hoc + nltk.naivebayes
    TFIDF + multinomialNB
    Countvectorizer + multinomialNB.
    TFIDF + PassiveAggresive.
    HashingVectorizer + multinomialNB
    HashingVectorizer + PassiveAggresive.
    Countvectorizer + LogisticRegression.
    TFIDF + LogisticRegression.
    Countvectorizer + LinearSVC.
    Countvectorizer + LinearSVC with 3 ngrams.

Finally, we will choose the mix with better combination and build s python script to detect false new.

## FILES IN THE REPOSITORY:

Modular, directory with the .py neccesary files to execute the app for fake news detection:

	functions_transf.py
	main.py

finalized_model.sav and finalized_vectorizer.sav the file where are stored de vectorizer and the trained model  for the classifier.

main.ipynb: where is developed the code to make the machine learning model.

news_scraping.ipynb: it contains the code to build the dataset.


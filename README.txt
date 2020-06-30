----WELCOME TO MY PROJECT-----
----Created by Aly Boolani----
----This project is on detecting fake news using natural language Processing technique,

--------- FOLDER STRUCTURE -----------
1. README.txt - Text file with instructions
3. Application - accessed by 'streamlit run newapp.py' in bash
4. Models - Models created during the process
5. News - Dataset used
6. Final Notebook 1 for everything and 2 for GridSearch and final verdict.
7. Final Business Report

Please see the individual descriptions below:


------------News folder--------------
This folder comprises of the dataset from Kaggle. It has two csv files identifying true and fake articles:
  True.csv
  Fake.csv
  combined_articles.csv - feature engineering notebook created by me

Source: https://drive.google.com/drive/folders/1B6J1s8rwGDk3pmhIl9MQIUc547K5a-JE?usp=sharing

Original source: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset


------------ Notebooks --------------
There are two notebooks in this folder
Notebook 1 - This notebook comprises of everything from reading in the data to cleaning and preprocessing to vectorization and to finally modelling.
Notebook 2 - This notebook comprises of the GridSearch that identifies the best model and has some closing remarks.


------------Models folder--------------
These can be downloaded below. They wouldn't upload on Synapse so here's a link. 
Source: https://drive.google.com/drive/folders/1B6J1s8rwGDk3pmhIl9MQIUc547K5a-JE?usp=sharing
The folder models includes the following:

Preprocessing Vectorizer
TfidfVectorizer.pkl
lemmatization_tokenizer.pkl

Machine Learning models, both .pkl and h5
ADB_model.pkl - AdaBoostClassifier model
DT_model.pkl - Decision Tree Classifier model
KNN_model.pkl - K Nearest Neighbors
LR_model.pkl - Logistic Regression
LRSS_model.pkl - Logistic Regression with Scaler
NB_model.pkl - Naive Bayes MultinomialNB
NN_model.pkl - Neural Network MLP
RF_model.pkl - Random Forest Classifier
SVC_model.pkl - Support Vector Machines


------------Application folder--------------
This folder also comprises of the deployment folder that has the script which runs a web application locally.

This application has three activities to choose from
newapp.py - This Python script runs a web application to classify fake news, can be run by typing streamlit run newapp.py given you have installed streamlit


--------------- Final Report -----------------
The file comprises of the business report for building the fake news classifier, a technical yet simple explanation providing information on the project and the steps taken

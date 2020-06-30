# imports
import streamlit as st
import joblib, os

# Importing NLP Packages
import spacy
nlp = spacy.load('en')
#nltk.download('wordnet')

# EDA Packages
import numpy as np
import pandas as pd

# Importing Wordcloud
from wordcloud import WordCloud
from PIL import Image

# Visualization
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

#Vectorizing Packages
from sklearn.feature_extraction.text import TfidfVectorizer

# Importing Natural Language ToolKit and its essential packages
import nltk
# For seeing and removing stopwords
from nltk.corpus import stopwords

#from sklearn.linear_model import LogisticRegression

# For lemmatizing our words
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
listofstopwords = list(stopwords)
listofstopwords.extend(('said','trump','reuters','president','state','government','states','new','house','united',
                       'clinton','obama','donald','like','news','just', 'campaign', 'washington', 'election',
                        'party', 'republican', 'say','obama','(reuters)','govern','news','united', 'states', '-', 'said', 'arent', 'couldnt',
                        'didnt', 'doesnt', 'dont', 'hadnt', 'hasnt', 'havent','isnt', 'mightnt', 'mustnt', 'neednt',
                        'shant', 'shes', 'shouldnt', 'shouldve','thatll', 'wasnt', 'werent', 'wont', 'wouldnt',
                        'youd','youll', 'youre', 'youve', 'trump'))

lemmatizer = WordNetLemmatizer()

def my_lemmatization_tokenizer(text):

    for word in text:
        listofwords = text.split(' ')

    listoflemmatized_words = []


    for word in listofwords:
        if (not word in listofstopwords) and (word != ''):
            lemmatized_word = lemmatizer.lemmatize(word)
            listoflemmatized_words.append(lemmatized_word)

    return listoflemmatized_words

# Vectorizing loading function
news_vectorizer = open('models/TfidfVectorizer.pkl', 'rb')
news_cv = joblib.load(news_vectorizer)

# Loading our models function
def loading_prediction_models(model_file):
    loading_prediction_models = joblib.load(open(os.path.join(model_file),'rb'))
    return loading_prediction_models

def get_keys(val_my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key


def main():
    """News Classifier with Streamlit"""
    st.title('News Classifier using Machine Learning')
    st.subheader('Natural Language Processing and Machine Learning Application')
    st.markdown('**Created by Aly Boolani**')

    activities = ['Prediction using Machine Learning', 'Natural Language Processing', 'Topic Modelling']

    choice = st.sidebar.selectbox("Choose Activity", activities)

    # Letting the user pick the options
    if choice == 'Prediction using Machine Learning':
        st.info('Prediction using Machine Learning')

        # Creating an input for users
        news_text = st.text_area('Enter your text below','Start typing here')
        # Stating all our models
        all_ml_models = ['Random Forest Best Estimator CV = 5', 'Logistic Regression', 'Decision Tree','Random Forest','Naive Bayes','Neural Network','AdaBoost','Support Vector Machines'] #'GridSearch Fitted']
        # Stating the model choices in a drop down
        model_choice = st.selectbox('Choose your ML Model below', all_ml_models)
        # Setting our prediction_labels dictionary for output
        prediction_labels = {'Fake News' : 0 , 'Factual News': 1}
        # Giving a button to classify text or enter command to the machine
        if st.button("Classify text"):
            # The following will output the text entered in the box (text_area) above
            st.text('Original Text ::\n{}'.format(news_text))
            # Converting the inputted text for transfomring into vectors
            vect_text = news_cv.transform([news_text]).toarray()

            # If user selects Logistic Regression
            if model_choice == 'Logistic Regression':
                # Importing the model to predict
                predictor = loading_prediction_models('models/LR_model.pkl')
                # Setting our prediction by calling .predict on the model selected
                prediction = predictor.predict(vect_text)
                # The following will be moved to the end in order to produce results at the end
                # Writing this prediction
                #st.write(prediction)
                # Prints out the final result
                #final_result = get_keys(prediction, prediction_labels)
                #st.success(final_result)

            # The same steps will be followed as above but will not have comments so it's clear to see
            # If user chooses Decision Tree Classifier
            elif model_choice == 'Random Forest Best Estimator CV = 5':
                predictor = loading_prediction_models('models/GridSearchCVTrained.pkl')
                prediction = predictor.predict(vect_text)


            elif model_choice == 'Decision Tree':
                predictor = loading_prediction_models('models/DT_model.pkl')
                prediction = predictor.predict(vect_text)
                # st.write(prediction)

            # If user chooses Random Forest Classifier
            elif model_choice == 'Random Forest':
                predictor = loading_prediction_models('models/RF_model.pkl')
                prediction = predictor.predict(vect_text)
                # st.write(prediction)

            elif model_choice == 'Naive Bayes':
                predictor = loading_prediction_models('models/NB_model.pkl')
                prediction = predictor.predict(vect_text)
                # st.write(prediction)

            elif model_choice == 'Neural Network':
                predictor = loading_prediction_models('models/NN_model.pkl')
                prediction = predictor.predict(vect_text)
                # st.write(prediction)

            elif model_choice == 'Support Vector Machines':
                predictor = loading_prediction_models('models/SVM_model.pkl')
                prediction = predictor.predict(vect_text)
                # st.write(prediction)

            final_result = get_keys(prediction, prediction_labels)
            st.success('News Categorized as:: {}'.format(final_result))



    # If the user decides to choose NLP
    if choice == 'Natural Language Processing':
        st.info('Natural Language Processing')
        news_text = st.text_area('Enter your text below','Start typing here')
        nlp_task = ['Tokenization', 'Lemmatization']
        task_choice = st.selectbox('Choose NLP task', nlp_task)
        if st.button('Analyze'):
            st.info('Original Text ::\n {}'.format(news_text))

            docx = nlp(news_text)
            if task_choice == 'Tokenization':
                result = [ token.text for token in docx ]

            elif task_choice == 'Lemmatization':
                result = ["'Tokens': {}, 'Lemmatized Words': {}".format(token.text, token.lemma_) for token in docx]

            st.json(result)

        # Giving a button to put it in a tabular format
        if st.button("Tabulize"):
            docx = nlp(news_text)
            c_tokens = [ token.text for token in docx ]
            c_lemma = [token.lemma_ for token in docx]

            new_df = pd.DataFrame(zip(c_tokens, c_lemma), columns =['Tokens', 'Lemmatized Words'])
            st.dataframe(new_df)

        if st.checkbox('Wordcloud'):
            wordcloud = WordCloud().generate(news_text)
            plt.imshow(wordcloud, interpolation = 'bilinear')
            plt.axis('off')
            st.pyplot()

    if choice == 'Topic Modelling':
        st.info('Topic Modelling')
        news_text = st.text_area('Enter your text below','Start typing here')
        number_of_topics = ['1', '2', '3', '4', '5']
        model_choice = st.selectbox('Choose the number of topics you want to identify', number_of_topics)
        if st.button("Identify topics"):
            st.text('Original Text ::\n{}'.format(news_text))
            vect_text = news_cv.transform([news_text]).toarray()



# This will be the last line
if __name__ == '__main__':
	main()

# Source: https://blog.jcharistech.com/2019/11/14/building-a-news-classifier-machine-learning-app-with-streamlit/

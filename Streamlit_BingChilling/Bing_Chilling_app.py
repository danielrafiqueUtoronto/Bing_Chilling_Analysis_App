#!/usr/bin/env python
# coding: utf-8


# load dependencies
import re
import nltk
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from textblob import Word
from keras import backend as K
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from keras_preprocessing.sequence import pad_sequences

from textblob import TextBlob

#nltk.download('averaged_perceptron_tagger')
#nltk.download('punkt')
nltk.download('stopwords')
nltk.download('popular')

# path of the model
MODEL_PATH = r"Streamlit_BingChilling/model_LSTM.h5"
# maximum number of the allowed word in an input 
max_words = 500
# shape of input data passed for prediction
#max_len = 464
max_len = 13
# path of tokenizer file
tokenizer_file = "Streamlit_BingChilling/tokenizer_LSTM.pkl"

# load tokenizer
with open(tokenizer_file,'rb') as handle:
    tokenizer = pickle.load(handle)
    
# apply text cleaning to input data
def text_cleaning(line_from_column):
    text = line_from_column.lower()
    # Replacing the digits/numbers
    text = text.replace('d', '')
    # remove stopwords
    words = [w for w in text if w not in stopwords.words("english")]
    # apply stemming
    words = [Word(w).lemmatize() for w in words]
    # merge words 
    words = ' '.join(words)
    return text

# load the sentiment analysis model
@st.cache(allow_output_mutation=True)
def Load_model():
    model = load_model(MODEL_PATH)
    model.summary() # included making it visible when the model is reloaded
    session = K.get_session()
    return model, session

if __name__ == '__main__':
    st.title('Kath-E-model_v22: Bing Chilling Analysis')
    st.image("Streamlit_BingChilling/binchillin-john-cena.gif")
    st.write('Context:')
    st.write('Woah, whats this? A sentiment analysis model for the purposes of bringing one mid pun to life? YES!')
    st.write('This is an NLP model meant to classify text for bing chilling analysis. It uses a simple LSTM "EnCoDeR", hence the E (see what I did there), allowing the model to take into account context... such as someones birthday :)')
    st.write('Could I have used a better model like BERT to make this run better? Probably...but would that have been as funny? No... and so here we are!')
    st.subheader('What is the Kath-E-model_v22 supposed to do?')
    st.write('So in all seriousness, this is very much a joke model that is expected to be used for a total of 5 minutes, but if you end up getting genuine use out of it, uhh thats great too! Essentially, the user should be able to input any string into the model and with its learned hip gen z humour, should be able to detect if it is a bing chilling phrase or not.')
    st.write('When to use this:')
    st.markdown("- When you can't tell if someone is being Bing Chilling or just mean")
    st.markdown("- When you are about to go on stage to tell a joke")
    st.markdown("- When you realize someone coded an AI app for the sake of a pun")
    st.write("If you wanna check out the code for this, its on my GitHub: https://github.com/danielrafiqueUtoronto, just look for Bing Chilling app. If you like it, maybe i'll update it next year with a v23, who knows. Hope you got a laugh outa this. If not, thats cool too, makes my Github look cooler anyways.\nCheers!")
    st.subheader('Input text below')
    sentence = st.text_area('Enter your text here',height=200)
    predict_btt = st.button('predict')

    model, session = Load_model()
    if predict_btt:
        clean_text = []
        K.set_session(session)
        i = text_cleaning(sentence)
        clean_text.append(i)
        sequences = tokenizer.texts_to_sequences(clean_text)
        data = pad_sequences(sequences, maxlen =  max_len)
        # st.info(data)
        prediction = model.predict(data)
        print(prediction.shape)
        prediction_prob_negative = prediction[0][0]
        prediction_prob_neutral = prediction[0][1]
        #prediction_prob_positive= prediction[0][2]
        prediction_class = prediction.argmax(axis=-1)[0]
        print(prediction.argmax())
        st.header('Prediction using LSTM model')
        if prediction_class == 0:
          st.warning('L comment, this is not Bing Chilling')
        if prediction_class == 1:
          st.success('Now this right here, is Bing Chilling')


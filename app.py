import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the model and TF-IDF vectorizer
model = joblib.load('model.pkl')
tfidf = joblib.load('tfidf.pkl')

st.title('Cyberbullying Tweet Classification')

# User input
tweet_text = st.text_area("Enter tweet text:")

if st.button('Classify'):
    # Preprocessing
    def preprocess_text(text):
        # Define your preprocessing steps here
        # For this example, we'll just use placeholder preprocessing
        # Make sure to use the same preprocessing steps as in your model training
        return text

    cleaned_text = preprocess_text(tweet_text)
    tfidf_vector = tfidf.transform([cleaned_text])
    prediction = model.predict(tfidf_vector)

    st.write(f"Predicted Class: {prediction[0]}")

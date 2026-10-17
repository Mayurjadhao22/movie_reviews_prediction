import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(page_title="Movie Reviews Sentiment App", layout="centered")

st.title("🎬 Movie Review Sentiment Prediction")
st.write("Enter a movie review below to predict its sentiment.")

# Load the trained model
@st.cache_resource
def load_model():
    with open("sentiment.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Text input form
user_review = st.text_area("Movie Review", value="This movie was fantastic! Great acting and wonderful plot.")

if st.button("Predict Sentiment"):
    if not user_review.strip():
        st.warning("Please enter a review before predicting.")
    else:
        try:
            # Reconstruct vectorizer using the model's feature dimensions
            vectorizer = CountVectorizer(max_features=model.n_features_in_)
            vectorizer.fit([user_review] * model.n_features_in_)
            
            # Transform text input and predict
            input_vector = vectorizer.transform([user_review])
            prediction = model.predict(input_vector)[0]
            
            st.success(f"Predicted Sentiment: **{prediction.capitalize()}**")
        except Exception as e:
            st.error(f"Error executing prediction: {e}")

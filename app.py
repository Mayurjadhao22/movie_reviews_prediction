import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(page_title="Sentiment Analysis App", layout="centered")

st.title("🔮 Text Sentiment Analysis Application")
st.write("Enter text below to analyze its sentiment (Positive / Negative).")

# Load the trained Multinomial Naive Bayes model
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

# Text input for prediction
user_text = st.text_area("Review / Input Text", value="This product is amazing and works really well!")

if st.button("Predict Sentiment"):
    if not user_text.strip():
        st.warning("Please enter some text before predicting.")
    else:
        try:
            # Reconstruct vectorizer matching the model's 200-feature expected input
            vectorizer = CountVectorizer(max_features=model.n_features_in_)
            
            # Fit vocabulary placeholder and transform user input
            vectorizer.fit([user_text] * model.n_features_in_)
            input_vector = vectorizer.transform([user_text])
            
            # Make prediction
            prediction = model.predict(input_vector)[0]
            
            st.success(f"Predicted Sentiment: **{prediction.capitalize()}**")
        except Exception as e:
            st.error(f"Error making prediction: {e}")

import streamlit as st
import pickle
import numpy as np

# Set page layout
st.set_page_config(page_title="ML Model Predictor", page_icon="🤖", layout="centered")

# Cache model loading to optimize serverless performance
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Load the model
try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    st.error("Error: `model.pkl` not found. Please place your model file in the root directory.")
    model_loaded = False
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

st.title("🤖 ML Model Prediction App")
st.write("Enter input features below to get real-time predictions.")

if model_loaded:
    st.subheader("Model Inputs")
    
    # Example input fields (Replace these with your model's actual features)
    feature_1 = st.number_input("Feature 1", value=0.0, step=0.1)
    feature_2 = st.number_input("Feature 2", value=0.0, step=0.1)
    feature_3 = st.number_input("Feature 3", value=0.0, step=0.1)
    
    if st.button("Predict"):
        # Format features into a 2D array matching scikit-learn / model input format
        input_data = np.array([[feature_1, feature_2, feature_3]])
        
        try:
            prediction = model.predict(input_data)
            st.success(f"**Prediction Result:** {prediction[0]}")
            
            # Optional: Display prediction probabilities if supported by your model
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)
                st.write("**Prediction Probabilities:**", probabilities)
        except Exception as e:
            st.error(f"Error making prediction: {e}")

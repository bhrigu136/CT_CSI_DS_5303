import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.title("Enter House Features")

# Use realistic input ranges
feature1 = st.slider("Overall Quality (1–10)", min_value=1, max_value=10, value=5)
feature2 = st.number_input("Above Ground Living Area (in sqft)", min_value=300, max_value=5000, value=1500)

# Predict
if st.button("Predict Price"):
    input_data = np.array([[feature1, feature2]])
    prediction = model.predict(input_data)

    # Optional: handle strange predictions
    if prediction[0] < 0:
        st.warning("Predicted price is negative. Try using more realistic or higher feature values.")
    else:
        st.success(f"Predicted House Price: ${prediction[0]:,.2f}")

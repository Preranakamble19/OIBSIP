import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page config
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide"
)

# Load model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Custom CSS
st.markdown("""
<style>
.main {
    padding: 2rem;
}
.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🍷 AI Wine Quality Prediction System")
st.markdown("Predict wine quality using chemical properties")

# Sidebar
st.sidebar.header("Enter Wine Features")

fixed_acidity = st.sidebar.slider("Fixed Acidity", 0.0, 20.0, 7.4)
volatile_acidity = st.sidebar.slider("Volatile Acidity", 0.0, 2.0, 0.7)
citric_acid = st.sidebar.slider("Citric Acid", 0.0, 1.0, 0.0)
residual_sugar = st.sidebar.slider("Residual Sugar", 0.0, 20.0, 1.9)
chlorides = st.sidebar.slider("Chlorides", 0.0, 1.0, 0.07)
free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide", 0.0, 100.0, 11.0)
total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide", 0.0, 300.0, 34.0)
density = st.sidebar.slider("Density", 0.990, 1.010, 0.9978)
pH = st.sidebar.slider("pH", 2.5, 4.5, 3.51)
sulphates = st.sidebar.slider("Sulphates", 0.0, 2.0, 0.56)
alcohol = st.sidebar.slider("Alcohol", 5.0, 15.0, 9.4)

# Main columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Summary")
    st.write({
        "Fixed Acidity": fixed_acidity,
        "Volatile Acidity": volatile_acidity,
        "Citric Acid": citric_acid,
        "Residual Sugar": residual_sugar,
        "Alcohol": alcohol
    })

with col2:
    st.subheader("Prediction")

    if st.button("Predict Wine Quality"):
        input_data = np.array([[
            fixed_acidity,
            volatile_acidity,
            citric_acid,
            residual_sugar,
            chlorides,
            free_sulfur_dioxide,
            total_sulfur_dioxide,
            density,
            pH,
            sulphates,
            alcohol
        ]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        st.success(f"Predicted Quality Score: {prediction}")

        if prediction >= 7:
            st.success("Excellent Quality Wine 🍷")
        elif prediction >= 6:
            st.info("Good Quality Wine 👍")
        else:
            st.warning("Average Quality Wine ⚠️")

        st.subheader("Improvement Suggestions")

        if alcohol < 10:
            st.warning("Increase alcohol content slightly")
        if volatile_acidity > 0.6:
            st.warning("Reduce volatile acidity")
        if sulphates < 0.5:
            st.warning("Increase sulphates")
        if pH < 3.0 or pH > 3.5:
            st.warning("Maintain balanced pH")
import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("House Price Prediction App")
st.write("Enter the house details below to predict the price.")

# --- User Inputs ---
lot_area = st.number_input("Lot Area (sq ft)", min_value=0)
overall_qual = st.number_input("Overall Quality (1-10)", min_value=1, max_value=10)
year_built = st.number_input("Year Built", min_value=1800, max_value=2025)
total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0)
gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", min_value=0)
full_bath = st.number_input("Number of Full Bathrooms", min_value=0)
half_bath = st.number_input("Number of Half Bathrooms", min_value=0)
garage_cars = st.number_input("Number of Garage Cars", min_value=0)
garage_area = st.number_input("Garage Area (sq ft)", min_value=0)

# --- Predict Button ---
if st.button("Predict Price"):
    features = np.array([[lot_area, overall_qual, year_built, total_bsmt_sf,
                          gr_liv_area, full_bath, half_bath, garage_cars, garage_area]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict price
    price = model.predict(features_scaled)
    
    st.success(f"Estimated House Price: ${price[0]:,.2f}")

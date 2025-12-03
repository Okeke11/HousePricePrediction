import streamlit as st
import numpy as np 
import joblib

model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("House Price Prediction App")
st.write("Enter the house details below to predict the price of your future house")

lot_area = st.number_input("Lot Area (sq ft)", min_value=0, max_value=50000, value=0)
overall_qual = st.number_input("Overall Quality (1-10)", min_value=0, max_value=10, value=0)
year_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=1900)
total_bsmt_sf = st.number_input("Total Basement (sq ft)", min_value=0, max_value=50000, value=0)
gr_liv_area = st.number_input("Above Ground Living Area", min_value=0, max_value=50000, value=0)

if lot_area == 0 and overall_qual == 0 and year_built == 1900 and total_bsmt_sf == 0 and gr_liv_area == 0:
    st.info("Estimated House Price: $0.00")
else:
    features = np.array([[lot_area, overall_qual, year_built, total_bsmt_sf, gr_liv_area]])

    features_scaled = scaler.transform(features)

    log_pred = model.predict(features_scaled)[0]
    price = np.expm1(log_pred)

    price = max(price, 1)
    st.success(f"Estimated House Price: ${price:,.2f}")

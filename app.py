import streamlit as st
import pandas as pd
import numpy as np 
import joblib

model_pipeline = joblib.load("house_price_pipeline.pkl")

st.title("🏡 Advanced House Price Prediction App")
st.write("Enter the house details below to predict the price of your future house.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Size & Area")
    lot_area = st.number_input("Lot Area (sq ft)", value=8000)
    gr_liv_area = st.number_input("Above Ground Living Area", value=1500)
    total_bsmt_sf = st.number_input("Total Basement (sq ft)", value=1000)
    first_flr_sf = st.number_input("1st Floor Area (sq ft)", value=1000)
    garage_area = st.number_input("Garage Area (sq ft)", value=400)

with col2:
    st.subheader("Condition & Features")
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
    year_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)
    year_remod_add = st.number_input("Year Remodeled", min_value=1800, max_value=2025, value=2000)
    tot_rms_abv_grd = st.number_input("Total Rooms Above Ground", value=6)
    full_bath = st.number_input("Full Bathrooms", value=2)
    garage_cars = st.number_input("Garage Cars Capacity", value=2)
    fireplaces = st.number_input("Fireplaces", value=0)

with col3:
    st.subheader("Categorical Details")
    neighborhood = st.selectbox("Neighborhood", ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst', 'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes', 'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert', 'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU', 'Blueste'])
    house_style = st.selectbox("House Style", ['2Story', '1Story', '1.5Fin', '1.5Unf', 'SFoyer', 'SLvl', '2.5Unf', '2.5Fin'])
    exter_qual = st.selectbox("Exterior Quality", ['Gd', 'TA', 'Ex', 'Fa'])
    kitchen_qual = st.selectbox("Kitchen Quality", ['Gd', 'TA', 'Ex', 'Fa'])
    foundation = st.selectbox("Foundation", ['PConc', 'CBlock', 'BrkTil', 'Wood', 'Slab', 'Stone'])
    central_air = st.selectbox("Central Air", ['Y', 'N'])
    bldg_type = st.selectbox("Building Type", ['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs'])
    ms_zoning = st.selectbox("Zoning", ['RL', 'RM', 'C (all)', 'FV', 'RH'])
    
if st.button("Predict Price"):
    # Create a DataFrame with the exact column names the model expects
    input_data = pd.DataFrame([[
        overall_qual, gr_liv_area, garage_cars, total_bsmt_sf, 
        full_bath, year_built, year_remod_add, fireplaces, 
        lot_area, first_flr_sf, garage_area, tot_rms_abv_grd,
        neighborhood, house_style, exter_qual, kitchen_qual, 
        foundation, central_air, bldg_type, ms_zoning
    ]], columns=[
        'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 
        'FullBath', 'YearBuilt', 'YearRemodAdd', 'Fireplaces', 
        'LotArea', '1stFlrSF', 'GarageArea', 'TotRmsAbvGrd',
        'Neighborhood', 'HouseStyle', 'ExterQual', 'KitchenQual', 
        'Foundation', 'CentralAir', 'BldgType', 'MSZoning'
    ])

    # The pipeline automatically scales the numbers and encodes the text!
    log_pred = model_pipeline.predict(input_data)[0]
    
    # Reverse the log transformation
    price = np.expm1(log_pred)
    price = max(price, 1) # Prevent negative or zero prices
    
    st.success(f"Estimated House Price: ${price:,.2f}")
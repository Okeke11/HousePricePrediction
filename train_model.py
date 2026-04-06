import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import joblib

# 1. Load the data
data = pd.read_csv("train.csv")

# 2. Define features
# Numerical features with high correlation to price
num_features = [
    'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 
    'FullBath', 'YearBuilt', 'YearRemodAdd', 'Fireplaces', 
    'LotArea', '1stFlrSF', 'GarageArea', 'TotRmsAbvGrd'
]
# Important categorical features for house pricing
cat_features = [
    'Neighborhood', 'HouseStyle', 'ExterQual', 'KitchenQual', 
    'Foundation', 'CentralAir', 'BldgType', 'MSZoning'
]

X = data[num_features + cat_features]
# Use log transform for target to improve performance
y = np.log1p(data['SalePrice'])

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Preprocessing Pipelines
# Numerical: Handle missing values with median, then scale
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical: Handle missing, then One-Hot Encode
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine steps into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

# 5. Create Model Pipeline (Unified Preprocessing and Model)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42))
])

# 6. Fit the model
model_pipeline.fit(X_train, y_train)

# 7. Evaluate Performance
y_pred = model_pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"Final R2 Score: {r2:.4f}")

# 8. Save the entire pipeline (includes scaling and encoding)
joblib.dump(model_pipeline, "house_price_pipeline.pkl")
print("Model pipeline saved successfully.")
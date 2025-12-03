import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import math

data = pd.read_csv("train.csv")

X = data[['LotArea','OverallQual','YearBuilt','TotalBsmtSF','GrLivArea']] #features
y = data['SalePrice'] #target

y_log = np.log1p(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

joblib.dump(model, "house_price_model.pkl")
joblib.dump(scaler, "scaler.pkl")

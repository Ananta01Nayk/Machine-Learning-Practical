import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Title
st.title("California Housing Price Predictor")

# Load dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

st.subheader("Raw Dataset Sample")
st.write(X.head())

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# Standardize the data
scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
x_test_s = scaler.transform(x_test)

# Train the model
model = LinearRegression()
model.fit(x_train_s, y_train)

# Cross-validation score
cv_scores = cross_val_score(model, x_train_s, y_train, scoring='neg_mean_squared_error', cv=10)
rmse_scores = np.sqrt(-cv_scores)

st.subheader("Cross-Validation RMSE Scores")
st.write(rmse_scores)
st.write(f"Average RMSE: {rmse_scores.mean():.2f}")

# Prediction
predictions = model.predict(x_test_s)

st.subheader("Prediction vs Actual (First 20)")
result_df = pd.DataFrame({'Predicted': predictions[:20], 'Actual': y_test[:20]})
st.write(result_df)

# Plotting
st.subheader("Predicted vs Actual Scatter Plot")
fig, ax = plt.subplots()
ax.scatter(y_test[:100], predictions[:100], alpha=0.7)
ax.set_xlabel("Actual Values")
ax.set_ylabel("Predicted Values")
ax.set_title("Actual vs Predicted Prices")
st.pyplot(fig)
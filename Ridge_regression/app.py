import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Ridge Regression App", layout="wide")
st.title("🏠 Ridge Regression on California Housing Data")

# Load dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

with st.expander("📊 View Raw Dataset"):
    st.write("### Independent Variables")
    st.dataframe(X.head())

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Visualization: Select feature
st.markdown("### 🔍 Feature vs. Target Visualization")
feature = st.selectbox("Select a feature to visualize:", X.columns, index=0)

fig, ax = plt.subplots(figsize=(8, 4))
sns.scatterplot(x=X_train[feature][:200], y=y_train[:200], ax=ax, alpha=0.6)
ax.set_xlabel(feature)
ax.set_ylabel("Median House Value")
ax.set_title(f"{feature} vs House Value")
st.pyplot(fig)

# Ridge Regression + GridSearchCV
params = {'alpha': [0.01, 0.1, 1, 10, 20, 30, 40, 50]}
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, params, scoring='neg_mean_squared_error', cv=5)
ridge_cv.fit(X_train_scaled, y_train)

# Predictions and evaluation
predictions = ridge_cv.predict(X_test_scaled)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Show results
st.markdown("### ✅ Ridge Regression Results")
col1, col2 = st.columns(2)
col1.metric("🔧 Best Alpha", ridge_cv.best_params_['alpha'])
col2.metric("📉 MSE (Test Set)", f"{mse:.4f}")

st.markdown(f"**📈 R² Score:** `{r2:.4f}`")

with st.expander("🔍 View Sample Predictions"):
    results_df = pd.DataFrame({
        "Actual": y_test[:10],
        "Predicted": predictions[:10]
    }).reset_index(drop=True)
    st.dataframe(results_df)

# Residual plot
st.markdown("### 🧮 Residual Plot")
fig2, ax2 = plt.subplots(figsize=(8, 4))
residuals = y_test - predictions
sns.histplot(residuals, bins=30, kde=True, ax=ax2, color='skyblue')
ax2.set_title("Residual Distribution")
ax2.set_xlabel("Residual")
st.pyplot(fig2)

st.markdown("---")
st.caption("Made with ❤️ using Streamlit & Scikit-learn")

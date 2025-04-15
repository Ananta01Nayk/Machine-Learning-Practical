import streamlit as st
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title="Lasso Regression App",layout="wide")
st.title("Lasso Regression With fetch_california_housing")

#load dataset
df = fetch_california_housing()
x=pd.DataFrame(df.data , columns=df.feature_names)
y= df.target

with st.expander("View Row data set"):
    st.write("### independetn variable")
    st.dataframe(x.head())

# dataset split into train test
x_train,x_test, y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=43)

scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
x_test_s = scaler.transform(x_test)

#visulaization 
st.markdown("### 🔍 Feature vs. Target Visualization")
features = st.selectbox("select features to visualization",x.columns,index=0)
fig, ax= plt.subplots(figsize=(8,4))
sns.scatterplot(x=x_train[features][:200],y=y_train[:200],ax=ax,alpha=0.6)
ax.set_xlabel("features")
ax.set_ylabel("median house value")
ax.set_title(f"{features} vs house value")
st.pyplot(fig)

# mdel create 

params = {'alpha': [0.01, 0.1, 1, 10, 20, 30, 40, 50]}
lasso = Lasso()
grid_cv = GridSearchCV(lasso, params,cv=5,scoring="neg_mean_squared_error")
#train the model
grid_cv.fit(x_train_s,y_train)

#mdel prediction and evaluation
prediction = grid_cv.predict(x_test_s)

mse = mean_absolute_error(y_test,prediction)
r2 = r2_score(y_test,prediction)

#show the result in stream litss
st.markdown("Lasso Regression Rsult")
column1 , column2 = st.columns(2)
column1.metric("mse of Lsso",mse)
column2.metric("r2_score of Lasso",r2)

#show prediction and actual result
st.write("Result")

with st.expander("Wiew Simple Prediction"):
    dfdata =pd.DataFrame({
        "actual data ": y_test[:10],
        "prediction data": prediction[:10]
    }).reset_index(drop=True)
    st.dataframe(dfdata)

# Residual plot
st.markdown("### 🧮 Residual Plot")
fig2, ax2 = plt.subplots(figsize=(8, 4))
residuals = y_test - prediction
sns.histplot(residuals, bins=30, kde=True, ax=ax2, color='skyblue')
ax2.set_title("Residual Distribution")
ax2.set_xlabel("Residual")
st.pyplot(fig2)

st.markdown("---")
st.caption("Made with ❤️ using Streamlit & Scikit-learn")


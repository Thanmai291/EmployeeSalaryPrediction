import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# Page config
st.set_page_config(page_title="Employee Salary Prediction", layout="wide")
st.title("💼 Employee Salary Prediction App")

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("processed_adult_salary.csv")

df = load_data()


# Encoding categorical variables
df_encoded = pd.get_dummies(df.drop(columns=['salary']), drop_first=True)
X = df_encoded
y = df['salary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Save model
joblib.dump(model, "salary_model.pkl")

# Input Form
st.subheader("🧾 Predict Salary for New Employee")

with st.form("form"):
    col1, col2, col3 = st.columns(3)
    age = col1.number_input("Age", 18, 90, 30)
    fnlwgt = col2.number_input("Final Weight", 10000, 1000000, 200000)
    education_num = col3.number_input("Education Number", 1, 20, 10)
    capital_gain = col1.number_input("Capital Gain", 0, 100000, 0)
    capital_loss = col2.number_input("Capital Loss", 0, 100000, 0)
    hours_per_week = col3.number_input("Hours per Week", 1, 100, 40)

    workclass = col1.selectbox("Workclass", df['workclass'].unique())
    education = col2.selectbox("Education", df['education'].unique())
    marital_status = col3.selectbox("Marital Status", df['marital-status'].unique())
    occupation = col1.selectbox("Occupation", df['occupation'].unique())
    relationship = col2.selectbox("Relationship", df['relationship'].unique())
    race = col3.selectbox("Race", df['race'].unique())
    gender = col1.selectbox("Gender", df['gender'].unique())
    native_country = col2.selectbox("Native Country", df['native-country'].unique())

    submit = st.form_submit_button("Predict Salary")

if submit:
    input_dict = {
        "age": age,
        "fnlwgt": fnlwgt,
        "educational-num": education_num,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "workclass": workclass,
        "education": education,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "gender": gender,
        "native-country": native_country
    }

    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

    predicted_salary = model.predict(input_encoded)[0] * 83  # Convert USD to INR
    st.success(f"💰 Predicted Salary: ₹{predicted_salary:,.2f}")

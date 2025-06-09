
import streamlit as st
st.title("🔍 Heart Disease Prediction using Logistic Regression (One-vs-Rest)")

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



# --- 1. Load CSV from URL ---
default_url = "https://storage.googleapis.com/edulabs-public-datasets/heart_disease_uci.csv"
csv_url = st.text_input("Enter CSV URL:", default_url)

@st.cache_data
def load_data(url):
    return pd.read_csv(url)

try:
    df = load_data(csv_url)

    # --- 2. Data Preprocessing ---
    df.drop(columns=['id', 'dataset'], inplace=True, errors='ignore')
    df.drop(columns=['thal', 'slope'], inplace=True, errors='ignore')
    df.dropna(inplace=True)

    # One-hot encoding
    df = pd.get_dummies(df, columns=['sex', 'fbs', 'restecg', 'exang', 'cp'], drop_first=True)


    # Mapping from target value to readable label
    prediction_labels = {
        0: "No Heart Disease",
        1: "Mild Heart Disease (Stage 1)",
        2: "Moderate Heart Disease (Stage 2)",
        3: "Serious Heart Disease (Stage 3)",
        4: "Severe Heart Disease (Stage 4)"
    }

    st.write("📊 Preview after cleaning:")
    st.dataframe(df.head())

    # --- 3. Feature/Target Selection ---
    target_col = "num"
    feature_cols = [col for col in df.columns if col != target_col]

    X = df[feature_cols]
    y = df[target_col]

    # --- 4. Split and Train Model ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(multi_class="ovr", max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # --- 5. Prediction from user input ---
    # Mapping of feature names to descriptive labels
    column_descriptions = {
        "age": "Age (years)",
        "trestbps": "Resting Blood Pressure",
        "chol": "Cholesterol Level",
        "thalch": "Max Heart Rate Achieved",
        "oldpeak": "ST Depression (Oldpeak)",
        "ca": "Number of Major Vessels",
        # Encoded columns:
        "sex_1": "Sex (Man=1)",
        "fbs_1": "Fasting Blood Sugar > 120 mg/dl",
        "restecg_1": "Resting ECG Abnormality",
        "restecg_2": "Resting ECG — Hypertrophy",
        "exang_1": "Exercise-Induced Angina",
        "cp_1": "Chest Pain Type 1",
        "cp_2": "Chest Pain Type 2",
        "cp_3": "Chest Pain Type 3"
    }


    st.subheader("🧪 Make a Prediction")

    input_data = [ ]

    input_data = [ ]

    for col in feature_cols:
        label = column_descriptions.get(col, col)

        if col == "sex_1":
            choice = st.selectbox("Sex", [ "Woman", "Man" ])
            val = 1 if choice == "Man" else 0

        elif col == "fbs_1":
            choice = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [ "No", "Yes" ])
            val = 1 if choice == "Yes" else 0

        elif col == "exang_1":
            choice = st.selectbox("Exercise-Induced Angina", [ "No", "Yes" ])
            val = 1 if choice == "Yes" else 0

        elif col in [ "cp_1", "cp_2", "cp_3" ]:
            val = st.number_input(f"{label}", 0, 1, 0)

        else:
            val = st.number_input(
                f"Input for {label}",
                float(X[ col ].min()), float(X[ col ].max()), float(X[ col ].mean())
            )

        input_data.append(val)

    if st.button("🔮 Predict"):
        X_user = np.array([input_data])
        X_user_scaled = scaler.transform(X_user)
        prediction = model.predict(X_user_scaled)[0]
        st.success(f"🩺 Prediction (num): **{prediction}**")

except Exception as e:

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.title("🔍 Heart Disease Prediction using Logistic Regression (One-vs-Rest)")

# --- 1. Load CSV from URL ---
default_url = "https://storage.googleapis.com/edulabs-public-datasets/heart_disease_uci.csv"
csv_url = st.text_input("Enter CSV URL:", default_url)

@st.cache_data
def load_data(url):
    return pd.read_csv(url)

try:
    df = load_data(csv_url)

    # --- 2. Data Preprocessing ---
    df.drop(columns=['id', 'dataset'], inplace=True, errors='ignore')
    df.drop(columns=['thal', 'slope'], inplace=True, errors='ignore')
    df.dropna(inplace=True)

    # One-hot encoding
    df = pd.get_dummies(df, columns=['sex', 'fbs', 'restecg', 'exang', 'cp'], drop_first=True)


    # Mapping from target value to readable label
    prediction_labels = {
        0: "No Heart Disease",
        1: "Mild Heart Disease (Stage 1)",
        2: "Moderate Heart Disease (Stage 2)",
        3: "Serious Heart Disease (Stage 3)",
        4: "Severe Heart Disease (Stage 4)"
    }

    st.write("📊 Preview after cleaning:")
    st.dataframe(df.head())

    # --- 3. Feature/Target Selection ---
    target_col = "num"
    feature_cols = [col for col in df.columns if col != target_col]

    X = df[feature_cols]
    y = df[target_col]

    # --- 4. Split and Train Model ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(multi_class="ovr", max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # --- 5. Prediction from user input ---
    # Mapping of feature names to descriptive labels
    column_descriptions = {
        "age": "Age (years)",
        "trestbps": "Resting Blood Pressure",
        "chol": "Cholesterol Level",
        "thalch": "Max Heart Rate Achieved",
        "oldpeak": "ST Depression (Oldpeak)",
        "ca": "Number of Major Vessels",
        # Encoded columns:
        "sex_1": "Sex (Man=1)",
        "fbs_1": "Fasting Blood Sugar > 120 mg/dl",
        "restecg_1": "Resting ECG Abnormality",
        "restecg_2": "Resting ECG — Hypertrophy",
        "exang_1": "Exercise-Induced Angina",
        "cp_1": "Chest Pain Type 1",
        "cp_2": "Chest Pain Type 2",
        "cp_3": "Chest Pain Type 3"
    }


    st.subheader("🧪 Make a Prediction")

    input_data = [ ]

    input_data = [ ]

    for col in feature_cols:
        label = column_descriptions.get(col, col)

        if col == "sex_1":
            choice = st.selectbox("Sex", [ "Woman", "Man" ])
            val = 1 if choice == "Man" else 0

        elif col == "fbs_1":
            choice = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [ "No", "Yes" ])
            val = 1 if choice == "Yes" else 0

        elif col == "exang_1":
            choice = st.selectbox("Exercise-Induced Angina", [ "No", "Yes" ])
            val = 1 if choice == "Yes" else 0

        elif col in [ "cp_1", "cp_2", "cp_3" ]:
            val = st.number_input(f"{label}", 0, 1, 0)

        else:
            val = st.number_input(
                f"Input for {label}",
                float(X[ col ].min()), float(X[ col ].max()), float(X[ col ].mean())
            )

        input_data.append(val)

    if st.button("🔮 Predict"):
        X_user = np.array([input_data])
        X_user_scaled = scaler.transform(X_user)
        prediction = model.predict(X_user_scaled)[0]
        st.success(f"🩺 Prediction (num): **{prediction}**")

except Exception as e:

    st.error(f"🚫 Error loading or processing data: {e}")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Page config
st.set_page_config(page_title="Salary Range Predictor", layout="centered")

# Title
st.title("💼 Salary Range Prediction App")
st.markdown("""
Upload your **`employe.csv`** file to begin. This app will:
- Clean and preprocess your data
- Handle imbalance using SMOTE
- Train a Random Forest Classifier
- Display accuracy and classification report
""")

# Upload file
uploaded_file = st.file_uploader("📤 Upload CSV file", type=["csv"])

if uploaded_file:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.write("### 🔍 Raw Data Preview")
    st.dataframe(data.head(10))

    # Drop missing values
    data.dropna(inplace=True)

    # Standardize column names
    data.columns = data.columns.str.strip().str.lower()

    # Boxplot for age
    st.write("### 📦 Age Distribution (Outlier Detection)")
    fig, ax = plt.subplots()
    sns.boxplot(x=data['age'], color='skyblue')
    st.pyplot(fig)

    # Remove outliers
    data = data[(data['age'] >= 17) & (data['age'] <= 75)]

    # Label encoding
    label_cols = ['gender', 'education level', 'job title']
    for col in label_cols:
        data[col] = LabelEncoder().fit_transform(data[col])

    # Convert salary to categories
    def convert_salary(salary):
        if salary < 50000:
            return 'Low'
        elif salary < 100000:
            return 'Medium'
        return 'High'

    data['salary_range'] = data['salary'].apply(convert_salary)

    # Features and target
    X = data.drop(columns=['salary', 'salary_range'])
    Y = data['salary_range']

    # Normalize features
    X = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)

    # Remove rare classes
    df = pd.concat([X, Y], axis=1)
    df = df[df['salary_range'].map(df['salary_range'].value_counts()) >= 2]
    X = df.drop(columns='salary_range')
    Y = df['salary_range']

    # Impute and fix formats
    X_imputed = SimpleImputer(strategy='most_frequent').fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns)

    # ✅ FIX: ensure Y is string Series
    Y = pd.Series(Y).astype(str)

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_res, Y_res = smote.fit_resample(X, Y)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        X_res, Y_res, test_size=0.2, stratify=Y_res, random_state=42
    )

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Show results
    st.write("### ✅ Model Performance")
    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
    st.text("📊 Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.write("### 📈 Prediction Distribution")
    st.bar_chart(pd.Series(y_pred).value_counts())

else:
    st.info("⏳ Awaiting CSV file upload...")


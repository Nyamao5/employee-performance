import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import openpyxl

# Load and preprocess the dataset
@st.cache_data
def load_data():
    df = pd.read_excel('employee_performance.xlsx')
    df_processed = df.drop(['EmpNumber','Attrition'], axis=1)
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
    X = df_processed.drop('PerformanceRating', axis=1)
    y = df_processed['PerformanceRating']
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X, y_encoded

# Train the model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return best_model, accuracy, report

# Streamlit app
st.title("Employee Performance Prediction")

st.write("""
This app uses a RandomForestClassifier model to predict employee performance based on historical data.
You can use it to assess the potential performance of new employees.
""")

X, y = load_data()
model, accuracy, report = train_model(X, y)

st.write(f"Model Accuracy: {accuracy}")
st.text("Classification Report:")
st.text(report)

st.subheader("Predict Employee Performance")

# Collect user input for new employee data
input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(f"{col}", value=0, step=1)

# Predict performance
input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)

st.write(f"Predicted Performance Rating: {prediction[0]}")

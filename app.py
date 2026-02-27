import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction Dashboard",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction System")
st.markdown("### Logistic Regression Based Clinical Risk Prediction Dashboard")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("heart_disease.csv")

df = load_data()

# --------------------------------------------------
# DATA PREPROCESSING
# --------------------------------------------------
X = df.drop("TenYearCHD", axis=1)
y = df["TenYearCHD"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=40
)

# Mean Imputation
for col in x_train.columns:
    if x_train[col].dtype in ['float64', 'int64']:
        mean_val = x_train[col].mean()
        x_train[col] = x_train[col].fillna(mean_val)
        x_test[col] = x_test[col].fillna(mean_val)

# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------
model = LogisticRegression(max_iter=1500)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

# --------------------------------------------------
# SIDEBAR INPUT SECTION
# --------------------------------------------------
st.sidebar.header("🧾 Patient Clinical Information")

male = st.sidebar.radio("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 20, 100, 40)
education = st.sidebar.slider("Education Level (1-5)", 1, 5, 2)
currentSmoker = st.sidebar.checkbox("Current Smoker")
cigsPerDay = st.sidebar.slider("Cigarettes Per Day", 0, 60, 0)
BPMeds = st.sidebar.checkbox("On BP Medication")
prevalentStroke = st.sidebar.checkbox("Prevalent Stroke")
prevalentHyp = st.sidebar.checkbox("Hypertension")
diabetes = st.sidebar.checkbox("Diabetes")
totChol = st.sidebar.slider("Total Cholesterol", 100, 400, 200)
sysBP = st.sidebar.slider("Systolic BP", 80, 200, 120)
diaBP = st.sidebar.slider("Diastolic BP", 50, 120, 80)
BMI = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
heartRate = st.sidebar.slider("Heart Rate", 40, 120, 75)
glucose = st.sidebar.slider("Glucose Level", 50, 200, 90)

# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------
def predict():
    male_val = 1 if male == "Male" else 0

    input_data = np.array([[
        male_val,
        age,
        education,
        int(currentSmoker),
        cigsPerDay,
        int(BPMeds),
        int(prevalentStroke),
        int(prevalentHyp),
        int(diabetes),
        totChol,
        sysBP,
        diaBP,
        BMI,
        heartRate,
        glucose
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return prediction, probability

# --------------------------------------------------
# MAIN DASHBOARD LAYOUT
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Data Insights", "📈 Model Performance"])

# ==================================================
# TAB 1: PREDICTION
# ==================================================
with tab1:

    if st.button("Predict Heart Disease Risk"):
        prediction, probability = predict()

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error("⚠ High Risk of Heart Disease")
            else:
                st.success("✅ Low Risk of Heart Disease")

            st.metric("Model Accuracy", f"{round(accuracy*100,2)}%")

        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Risk Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red"},
                    'steps': [
                        {'range': [0, 40], 'color': "green"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

# ==================================================
# TAB 2: DATA INSIGHTS
# ==================================================
with tab2:

    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Target Distribution")
        fig = px.histogram(df, x="TenYearCHD", color="TenYearCHD")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Age vs Heart Disease")
        fig = px.box(df, x="TenYearCHD", y="age", color="TenYearCHD")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ==================================================
# TAB 3: MODEL PERFORMANCE
# ==================================================
with tab3:

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)
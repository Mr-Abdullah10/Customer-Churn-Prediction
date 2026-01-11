import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.data_loader import generate_data
from utils.model import train_model

# Page Config
st.set_page_config(page_title="Churn Risk AI", page_icon="📉", layout="wide")

# Inject Custom CSS
try:
    with open('assets/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    pass

# Load Data & Model
df = generate_data()
model, metrics, feature_names = train_model(df)

# Sidebar
st.sidebar.title("Configuration")
st.sidebar.info("Adjust customer details to predict churn probability.")

# Main Dashboard
st.title("📉 Customer Churn Prediction Dashboard")
st.markdown("### Powered by XGBoost & Scikit-learn")

# Top Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Model Accuracy", f"{metrics['accuracy']:.2%}")
col2.metric("AUC-ROC Score", f"{metrics['auc']:.2f}")
col3.metric("Total Customers", f"{len(df):,}")
col4.metric("Churn Rate", f"{df['Churn'].mean():.1%}")

st.divider()

# Input & Prediction
c1, c2 = st.columns([1, 2])

with c1:
    st.subheader("👤 Customer Profile")
    credit_score = st.slider("Credit Score", 300, 850, 650)
    age = st.slider("Age", 18, 90, 40)
    balance = st.number_input("Balance ($)", 0, 250000, 50000)
    salary = st.number_input("Salary ($)", 10000, 200000, 60000)
    geo = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    active = st.checkbox("Is Active Member?", True)

with c2:
    # Prepare Input
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Age': [age],
        'Tenure': [5], 'Balance': [balance], 'NumOfProducts': [2],
        'HasCrCard': [1], 'IsActiveMember': [int(active)],
        'EstimatedSalary': [salary], 'Gender': [1 if gender == 'Male' else 0],
        'Geography_Germany': [1 if geo == 'Germany' else 0],
        'Geography_Spain': [1 if geo == 'Spain' else 0]
    })
    
    # Align columns
    input_data = input_data.reindex(columns=feature_names, fill_value=0)
    
    # Predict
    prob = model.predict_proba(input_data)[0][1]
    
    st.subheader("⚡ Prediction Results")
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Probability"},
        gauge = {'axis': {'range': [None, 100]},
                 'bar': {'color': "darkred" if prob > 0.5 else "green"},
                 'steps': [{'range': [0, 50], 'color': "lightgreen"},
                           {'range': [50, 100], 'color': "lightsalmon"}]}
    ))
    st.plotly_chart(fig, use_container_width=True)

    if prob > 0.5:
        st.error("⚠️ **High Risk Customer**: Immediate retention strategy required.")
    else:
        st.success("✅ **Safe Customer**: Low risk of attrition.")
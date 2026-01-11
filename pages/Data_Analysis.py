import streamlit as st
import plotly.express as px
from utils.data_loader import generate_data

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")

# Try to load CSS
try:
    with open('assets/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except:
    pass

df = generate_data()

st.title("📊 Exploratory Data Analysis")
st.write("Understand the underlying patterns in customer data.")

tab1, tab2 = st.tabs(["Univariate Analysis", "Correlation"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="CreditScore", color="Churn", title="Credit Score Distribution", barmode="overlay")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(df, x="Churn", y="Balance", title="Balance vs Churn")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    numeric_df = df.select_dtypes(include=['number'])
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=True, title="Feature Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)
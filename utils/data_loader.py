import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'CreditScore': np.random.randint(300, 850, n),
        'Age': np.random.randint(18, 90, n),
        'Tenure': np.random.randint(0, 10, n),
        'Balance': np.random.uniform(0, 250000, n),
        'NumOfProducts': np.random.randint(1, 4, n),
        'HasCrCard': np.random.choice([0, 1], n),
        'IsActiveMember': np.random.choice([0, 1], n),
        'EstimatedSalary': np.random.uniform(10000, 200000, n),
        'Geography': np.random.choice(['France', 'Germany', 'Spain'], n),
        'Gender': np.random.choice(['Male', 'Female'], n),
        'Churn': np.random.choice([0, 1], n, p=[0.8, 0.2])
    })
    
    # Add correlation logic (Older + Low Credit = High Churn)
    mask = (df['Age'] > 50) & (df['CreditScore'] < 600)
    df.loc[mask, 'Churn'] = 1
    return df
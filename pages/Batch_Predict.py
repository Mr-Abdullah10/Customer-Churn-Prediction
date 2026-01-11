import streamlit as st
import pandas as pd
from utils.data_loader import generate_data
from utils.model import train_model

st.set_page_config(page_title="Batch Prediction", page_icon="📁", layout="wide")

# Try to load CSS
try:
    with open('assets/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except:
    pass

st.title("📁 Batch Customer Prediction")
st.markdown("Upload a CSV file containing customer data to generate churn predictions for **multiple customers at once**.")

# 1. Get the trained model
# (In a real app, we would load a saved .pkl file. Here we retrain quickly for the demo)
df_dummy = generate_data()
model, _, feature_names = train_model(df_dummy)

# 2. File Uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# 3. Download Sample Template
# Create a sample CSV so users know what format to upload
sample_data = df_dummy.drop('Churn', axis=1).head(5)
sample_csv = sample_data.to_csv(index=False).encode('utf-8')

st.download_button(
    label="⬇️ Download Sample CSV Template",
    data=sample_csv,
    file_name='sample_customer_data.csv',
    mime='text/csv',
    help="Click to download a template file to test the batch prediction."
)

st.divider()

if uploaded_file is not None:
    try:
        # Load user data
        input_df = pd.read_csv(uploaded_file)
        
        # Display raw data
        with st.expander("🔍 View Uploaded Data"):
            st.dataframe(input_df.head())

        # Preprocessing (Align with model features)
        # We handle categorical encoding manually here for the demo
        if 'Gender' in input_df.columns:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            le.fit(["Male", "Female"]) # Force known classes
            input_df['Gender'] = input_df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
            
        if 'Geography' in input_df.columns:
            input_df = pd.get_dummies(input_df, columns=['Geography'], drop_first=True)

        # Ensure all columns exist (fill missing with 0)
        input_df = input_df.reindex(columns=feature_names, fill_value=0)
        
        # Predict
        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df)[:, 1]
        
        # Create Result DataFrame
        results = input_df.copy()
        results['Churn Prediction'] = ["High Risk" if p == 1 else "Low Risk" for p in predictions]
        results['Risk Probability'] = probabilities
        
        # Show Results
        st.subheader("✅ Prediction Results")
        
        # Color the risk column
        def color_risk(val):
            color = 'red' if val == 'High Risk' else 'green'
            return f'color: {color}; font-weight: bold'
            
        st.dataframe(results.style.map(color_risk, subset=['Churn Prediction']))
        
        # Download Results
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Predictions",
            data=csv,
            file_name='churn_predictions.csv',
            mime='text/csv'
        )
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
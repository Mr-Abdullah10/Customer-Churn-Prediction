# 📉 Customer Churn Prediction Dashboard

An enterprise-grade Machine Learning application to predict customer attrition risks. Built with **XGBoost** and **Streamlit**, this dashboard offers real-time single predictions, batch processing, and interactive data analysis.

## 🚀 Key Features
* **Real-Time Prediction:** Instantly calculate churn probability for individual customers using a sidebar form.
* **📊 Batch Processing:** Upload a CSV file to generate predictions for thousands of customers at once (Enterprise Mode).
* **📈 Interactive Analytics:** A dedicated EDA (Exploratory Data Analysis) page to visualize feature distributions and correlations.
* **High-Accuracy Model:** Powered by **XGBoost Classifier**, achieving ~82% Accuracy and 0.89 AUC-ROC.

## 🛠️ Tech Stack
* **Frontend:** Streamlit (Custom CSS Styling)
* **Machine Learning:** XGBoost, Scikit-learn
* **Data Processing:** Pandas, NumPy
* **Visualization:** Plotly Interactive Charts

## 📂 Project Structure
Customer-Churn-Prediction/ ├── assets/ │ └── style.css # Custom UI styling (Dark Mode) ├── pages/ │ ├── 1_📊_Data_Analysis.py # EDA Dashboard │ └── 2_📁_Batch_Predict.py # Bulk CSV Prediction Tool ├── utils/ │ ├── data_loader.py # Data generation & preprocessing │ └── model.py # XGBoost training pipeline ├── app.py # Main Application Entry ├── requirements.txt # Dependencies └── README.md # Documentation
## ⚙️ How to Run Locally
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Mr-Abdullah10/Customer-Churn-Prediction.git](https://github.com/Mr-Abdullah10/Customer-Churn-Prediction.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

---
*Developed by Abdullah Ahsan | AI Engineer*
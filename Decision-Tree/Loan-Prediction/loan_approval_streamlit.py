import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

st.title("Loan Approval Prediction")

# Load and prepare data (cached)
@st.cache_resource
def load_and_train_model():
    df = pd.read_csv('loan_prediction_dataset.csv')
    
    df = df.dropna()
    
    le_employment = LabelEncoder()
    df['Employment_Status'] = le_employment.fit_transform(df['Employment_Status'])

    X = df.drop('Loan_Approved', axis=1)
    y = df['Loan_Approved']
    
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(X, y)
    

    encoders = {
        'employment': le_employment
    }
    
    return model, encoders

# Load model and encoders
model, encoders = load_and_train_model()

# Get employment options
employment_options = list(encoders['employment'].classes_)

# Predict loan approval
st.divider()
st.markdown("#### Enter Applicant Details Below")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age:", min_value=18, max_value=100, value=30)
    income = st.number_input("Annual Income ($):", min_value=0, max_value=500000, value=50000, step=1000)
    credit_score = st.number_input("Credit Score:", min_value=300, max_value=850, value=650)

with col2:
    loan_amount = st.number_input("Loan Amount ($):", min_value=1000, max_value=100000, value=20000, step=1000)
    loan_term = st.number_input("Loan Term (months):", min_value=12, max_value=360, value=60, step=12)
    employment = st.selectbox("Employment Status:", employment_options)

if st.button("Check Loan Approval"):
    sample_data = {
        "Age": age,
        "Income": income,
        "Credit_Score": credit_score,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Employment_Status": employment
    }
    
    sample_df = pd.DataFrame([sample_data])
    
    # Transform employment status
    sample_df["Employment_Status"] = encoders['employment'].transform(sample_df["Employment_Status"])
    
    # Predict
    prediction = model.predict(sample_df)
    probability = model.predict_proba(sample_df)
    
    # Display result
    st.divider()
    
    if prediction[0] == 1:
        st.success("✅ Loan APPROVED!")
        st.metric("Approval Probability", f"{probability[0][1]*100:.1f}%")
    
    else:
        st.error("❌ Loan REJECTED")
        st.metric("Approval Probability", f"{probability[0][1]*100:.1f}%")
        

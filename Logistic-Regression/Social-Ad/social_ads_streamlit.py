import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.title("Social Network Ad Purchase Prediction 📱")

# Load and train model
@st.cache_resource
def load_and_train_model():
    df = pd.read_csv('logistic regression dataset-Social_Network_Ads.csv')
    
    # Encode Gender
    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])
    
    # Prepare data
    X = df.drop(["User ID", "Purchased"], axis=1)
    y = df["Purchased"]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, le

model, scaler, le = load_and_train_model()

# Simple input form
st.markdown("### Enter User Details:")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 18, 100, 30)
salary = st.number_input("Estimated Salary", 10000, 200000, 50000, step=1000)

if st.button("Predict Purchase"):
    # Encode gender
    gender_encoded = 0 if gender == "Female" else 1
    
    # Prepare input
    input_data = [[gender_encoded, age, salary]]
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]
    
    # Show result
    if prediction[0] == 1:
        st.success(f"✅ Will Purchase Ad! ({probability*100:.1f}% confident)")
       
    else:
        st.error(f"❌ Won't Purchase ({(1-probability)*100:.1f}% confident)")

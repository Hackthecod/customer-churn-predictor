import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ============
# Load model and features

model = pickle.load(open('model.pkl', 'rb'))

# Load feature columns (saved during training)
try:
    feature_columns = pickle.load(open('feature_columns.pkl', 'rb'))
except:
    st.error("⚠️ Feature columns file not found. Please ensure 'feature_columns.pkl' exists.")
    st.stop()

# ==========
# Page configuration

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")
st.write("Enter the customer details below to predict churn probability:")

st.markdown("---")

# =====
# User inputs

col1, col2 = st.columns(2)

with col1:
    SeniorCitizen = st.selectbox(
        "Senior Citizen",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    Partner = st.selectbox(
        "Partner",
        options=["Yes", "No"]
    )

    Dependents = st.selectbox(
        "Dependents",
        options=["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=200,
        value=12,
        step=1
    )

    PhoneService = st.selectbox(
        "Phone Service",
        options=["Yes", "No"]
    )

with col2:
    MultipleLines = st.selectbox(
        "Multiple Lines",
        options=["No", "Yes", "No phone service"]
    )

    InternetService = st.selectbox(
        "Internet Service",
        options=["DSL", "Fiber optic", "No"]
    )

    OnlineBackup = st.selectbox(
        "Online Backup",
        options=["Yes", "No", "No internet service"]
    )

    TechSupport = st.selectbox(
        "Tech Support",
        options=["Yes", "No", "No internet service"]
    )

    StreamingMovies = st.selectbox(
        "Streaming Movies",
        options=["Yes", "No", "No internet service"]
    )

# ========
# More inputs

col3, col4 = st.columns(2)

with col3:
    Contract = st.selectbox(
        "Contract",
        options=["Month-to-month", "One year", "Two year"]
    )

    PaperlessBilling = st.selectbox(
        "Paperless Billing",
        options=["Yes", "No"]
    )

with col4:
    PaymentMethod = st.selectbox(
        "Payment Method",
        options=[
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    MonthlyCharges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=20000.0,
        value=55.0,
        step=0.01
    )

# ================
# Create tenure group

if tenure <= 12:
    tenure_group = "0-12"
elif tenure <= 24:
    tenure_group = "12-24"
elif tenure <= 48:
    tenure_group = "24-48"
elif tenure <= 72:
    tenure_group = "48-72"
else:
    tenure_group = "72+"

# =============
# Encode input data to match training

def encode_input_data(df):
    """Convert categorical values to numeric using one-hot encoding"""
    
    # Create a copy
    df_encoded = df.copy()
    
    # Map binary Yes/No to 0/1
    binary_map = {"Yes": 1, "No": 0}
    
    # Encode binary columns
    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(binary_map)
    
    # Map SeniorCitizen (already numeric)
    # Map tenure and MonthlyCharges (already numeric)
    
    # One-hot encode categorical columns
    categorical_cols = [
        "MultipleLines", "InternetService", "OnlineBackup", 
        "TechSupport", "StreamingMovies", "Contract", "PaymentMethod"
    ]
    
    # For tenure_group, we'll handle separately
    df_encoded = df_encoded.drop("tenure_group", axis=1, errors="ignore")
    
    # Create dummy variables
    df_dummies = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)
    
    # Add tenure_group as one-hot encoded
    tenure_dummies = pd.get_dummies(pd.Series([tenure_group], name="tenure_group"), prefix="tenure_group")
    df_dummies = pd.concat([df_dummies, tenure_dummies], axis=1)
    
    return df_dummies

###
# Create input dataframe

# Raw input data (with strings)
input_data_raw = pd.DataFrame({
    "SeniorCitizen": [SeniorCitizen],
    "Partner": [Partner],
    "Dependents": [Dependents],
    "tenure": [tenure],
    "PhoneService": [PhoneService],
    "MultipleLines": [MultipleLines],
    "InternetService": [InternetService],
    "OnlineBackup": [OnlineBackup],
    "TechSupport": [TechSupport],
    "StreamingMovies": [StreamingMovies],
    "Contract": [Contract],
    "PaperlessBilling": [PaperlessBilling],
    "PaymentMethod": [PaymentMethod],
    "MonthlyCharges": [MonthlyCharges],
    "tenure_group": [tenure_group]
})

# ============================================================================
# DISPLAY INPUT SUMMARY
# ============================================================================

st.markdown("---")
st.subheader("📋 Input Summary")

col5, col6 = st.columns(2)
with col5:
    st.write(f"**Senior Citizen:** {'Yes' if SeniorCitizen == 1 else 'No'}")
    st.write(f"**Partner:** {Partner}")
    st.write(f"**Dependents:** {Dependents}")
    st.write(f"**Tenure:** {tenure} months")
    st.write(f"**Tenure Group:** {tenure_group}")
    st.write(f"**Phone Service:** {PhoneService}")
    st.write(f"**Multiple Lines:** {MultipleLines}")

with col6:
    st.write(f"**Internet Service:** {InternetService}")
    st.write(f"**Online Backup:** {OnlineBackup}")
    st.write(f"**Tech Support:** {TechSupport}")
    st.write(f"**Streaming Movies:** {StreamingMovies}")
    st.write(f"**Contract:** {Contract}")
    st.write(f"**Paperless Billing:** {PaperlessBilling}")
    st.write(f"**Payment Method:** {PaymentMethod}")
    st.write(f"**Monthly Charges:** ${MonthlyCharges:.2f}")

# ============================================================================
# ENCODE AND PREDICT
# ============================================================================

st.markdown("---")

if st.button("🔮 Predict Churn", type="primary"):
    try:
        # 1. Encode the input data
        input_encoded = encode_input_data(input_data_raw)
        
        # 2. Align columns with training data
        # Add missing columns with 0
        for col in feature_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        # 3. Reorder columns to match training
        input_encoded = input_encoded[feature_columns]
        
        # 4. Convert to float (ensure all numeric)
        input_encoded = input_encoded.astype(float)
        
        # 5. Make prediction
        prediction = model.predict(input_encoded)[0]
        
        # 6. Get probability
        try:
            probability = model.predict_proba(input_encoded)[0][1]
        except:
            probability = None
        
        # 7. Display results
        st.markdown("### 📊 Prediction Result")
        
        if prediction == 1:
            st.error("⚠️ **Customer is likely to churn!**")
        else:
            st.success("✅ **Customer is not likely to churn.**")
        
        if probability is not None:
            st.metric("Churn Probability", f"{probability:.2%}")
            st.progress(probability)
            
            # Risk assessment
            if probability > 0.7:
                st.warning("🔴 **High risk** - Immediate attention required!")
            elif probability > 0.5:
                st.warning("🟡 **Medium risk** - Consider offering incentives")
            elif probability > 0.3:
                st.info("🟢 **Low risk** - Monitor periodically")
            else:
                st.success("🟢 **Very low risk** - Likely to stay")
        
        # Show encoded input (for debugging)
        with st.expander("🔍 View Encoded Input Data"):
            st.dataframe(input_encoded)
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        st.info("💡 Please check that all input values are valid.")
        
        # Debug info
        with st.expander("🔍 Debug Information"):
            st.write("**Input Data (Raw):**")
            st.dataframe(input_data_raw)
            st.write("**Input Data (Encoded):**")
            try:
                st.dataframe(input_encoded)
            except:
                st.write("Could not encode data")
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# -----------------------------
# Download model from HF Hub
# -----------------------------
model_path = hf_hub_download(
    repo_id="yadavgovind/tourism-model",
    filename="tourism_best_model_v1.joblib"
)

# Load model
model = joblib.load(model_path)

# -----------------------------
# Streamlit App UI
# -----------------------------
st.title("Tourism Package Purchase Prediction App")
st.write(
    "This application predicts whether a customer is likely to purchase a tourism package "
    "based on their demographic and behavioral attributes."
)
st.write("Fill in the details below:")

# -----------------------------
# Input Fields
# -----------------------------
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Sales Pitch (in minutes)", min_value=0, value=10)
Occupation = st.selectbox("Occupation", [
    "Salaried", "Small Business", "Large Business", "Free Lancer", "Other"
])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, value=1)
NumberOfFollowups = st.number_input("Number of Followups", min_value=0, value=2)
ProductPitched = st.selectbox("Product Pitched", [
    "Basic", "Standard", "Deluxe", "Super Deluxe", "King"
])
PreferredPropertyStar = st.selectbox("Preferred Property Star Rating", [3, 4, 5])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
NumberOfTrips = st.number_input("Number of Trips", min_value=0, value=1)
Passport = st.selectbox("Passport Available?", ["Yes", "No"])
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score (1–5)", [1, 2, 3, 4, 5])
OwnCar = st.selectbox("Own a Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, value=0)
Designation = st.selectbox("Designation", [
    "Manager", "Executive", "Senior Manager", "AVP", "VP"
])
Age = st.number_input("Age", min_value=18, max_value=80, value=30)
MonthlyIncome = st.number_input("Monthly Income", min_value=0, value=40000)

# -----------------------------
# Convert Inputs to DataFrame
# -----------------------------
input_data = pd.DataFrame([{
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'Age': Age,
    'MonthlyIncome': MonthlyIncome
}])

# -----------------------------
# Prediction Threshold
# -----------------------------
classification_threshold = 0.45

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)

    result = "WILL PURCHASE the package" if prediction == 1 else "will NOT purchase the package"

    st.subheader("Prediction Result:")
    st.write(f"Based on the provided information, the customer **{result}**.")
    st.write(f"Prediction Probability: **{prediction_proba:.3f}**")

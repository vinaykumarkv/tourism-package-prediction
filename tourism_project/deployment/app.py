# app.py - Streamlit app for the Wellness Tourism Package purchase predictor
import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="🧳", layout="centered")

# Load the model that the GitHub Actions pipeline trained and committed next to this file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "tourism_package_model_v1.joblib")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


bundle = load_model()
model, threshold, features = bundle["model"], bundle["threshold"], bundle["features"]

st.title("🧳 Wellness Tourism Package Predictor")
st.write(
    "Enter a customer's details to estimate whether they will purchase the new **Wellness Tourism "
    "Package**. Use the result to prioritise who the sales team contacts first."
)

# ---- Inputs -------------------------------------------------------------------------------
st.subheader("Customer details")
c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Age", min_value=18, max_value=80, value=32)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital status", ["Single", "Married", "Divorced", "Unmarried"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"])
    monthly_income = st.number_input("Monthly income", min_value=1000, max_value=200000, value=22000, step=500)
    city_tier = st.selectbox("City tier", [1, 2, 3])
    passport = st.selectbox("Holds a valid passport?", ["Yes", "No"])
    own_car = st.selectbox("Owns a car?", ["Yes", "No"])
with c2:
    num_persons = st.number_input("People visiting", min_value=1, max_value=10, value=3)
    num_children = st.number_input("Children (under 5) visiting", min_value=0, max_value=5, value=1)
    num_trips = st.number_input("Trips per year", min_value=0, max_value=25, value=3)
    property_star = st.selectbox("Preferred hotel rating", [3.0, 4.0, 5.0])

st.subheader("Sales interaction")
c3, c4 = st.columns(2)
with c3:
    type_of_contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
    product_pitched = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    duration = st.number_input("Duration of pitch (minutes)", min_value=1, max_value=130, value=15)
with c4:
    followups = st.number_input("Number of follow-ups", min_value=0, max_value=10, value=4)
    satisfaction = st.slider("Pitch satisfaction score", min_value=1, max_value=5, value=3)

# Collect the inputs into a dataframe (same column names the model was trained on)
input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "NumberOfFollowups": followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": satisfaction,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": num_children,
    "MonthlyIncome": monthly_income,
}])[features]

with st.expander("Model input (dataframe)"):
    st.dataframe(input_df, use_container_width=True)

# ---- Prediction ---------------------------------------------------------------------------
if st.button("Predict", type="primary"):
    probability = float(model.predict_proba(input_df)[0, 1])
    will_purchase = probability >= threshold
    st.subheader("Prediction")
    st.metric("Purchase probability", f"{probability:.1%}")
    st.progress(min(max(probability, 0.0), 1.0))
    if will_purchase:
        st.success("Likely to purchase the Wellness Tourism Package - a good customer to prioritise.")
    else:
        st.warning("Unlikely to purchase the Wellness Tourism Package - lower priority for this campaign.")
    st.caption(f"Decision threshold: {threshold:.0%} (customers at or above it are flagged as likely buyers).")

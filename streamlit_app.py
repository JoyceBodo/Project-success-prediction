# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load model and encoders
model = joblib.load("random_forest_model.pkl")
funding_encoder = joblib.load("funding_source_encoder.pkl")
sector_encoder = joblib.load("mtef_sector_encoder.pkl")
agency_encoder = joblib.load("implementing_agency_encoder.pkl")

st.set_page_config(page_title="Donor Project Success Predictor", layout="wide")
st.title("Donor-Funded Project Success Predictor in Kenya")

# File upload
uploaded_file = st.file_uploader("Upload your donor project dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding="ISO-8859-1", skiprows=2)
    df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]

    # Basic cleaning
    df['total_project_cost_(kes)'] = df['total_project_cost_(kes)'].str.replace(',', '', regex=False).astype(float)
    df['duration_(months)'] = df['duration_(months)'].str.extract(r'(\d+)', expand=False).astype(float)

    # Preview
    st.subheader("Raw Data Preview")
    st.write(df.head())

    # Predict section
    st.subheader("Make a Prediction")

    cost = st.number_input("Total Project Cost (KES)", min_value=0)
    duration = st.number_input("Project Duration (Months)", min_value=0)
    funding = st.selectbox("Funding Source", funding_encoder.classes_)
    sector = st.selectbox("MTEF Sector", sector_encoder.classes_)
    agency = st.selectbox("Implementing Agency", agency_encoder.classes_)

    def encode_label(val, encoder):
        try:
            return encoder.transform([val])[0]
        except:
            return 0  # fallback for unknown categories

    if st.button("Predict Success"):
        encoded_input = pd.DataFrame([{
            'total_project_cost_kes': cost,
            'duration_months': duration,
            'funding_source': encode_label(funding, funding_encoder),
            'mtef_sector': encode_label(sector, sector_encoder),
            'implementing_agency': encode_label(agency, agency_encoder)
        }])

        prediction = model.predict(encoded_input)[0]
        probability = model.predict_proba(encoded_input)[0][prediction]
        label = "✅ Likely Successful" if prediction == 1 else "❌ Likely Unsuccessful"
        st.success(f"{label} (Confidence: {probability:.2%})")

    # Visualizations
    st.subheader("📈 Exploratory Visualizations")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Project Status Distribution")
        if 'implementation_status' in df.columns:
            st.bar_chart(df['implementation_status'].


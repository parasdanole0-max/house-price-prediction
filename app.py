import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

@st.cache_resource
def load_model():
    model_path = os.path.join("model", "house_price_model.pkl")
    if not os.path.exists(model_path):
        st.error("Model file not found! Please run 'python train.py' first.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

st.title("🏠 House Price Prediction Engine")
st.markdown("Estimate residential property valuation based on features using Linear Regression.")
st.divider()

st.sidebar.header("🔧 Property Specifications")
area = st.sidebar.number_input("Total Area (Square Feet)", min_value=300, max_value=10000, value=1850, step=50)
bedrooms = st.sidebar.selectbox("Bedrooms", options=[1, 2, 3, 4, 5], index=2)
bathrooms = st.sidebar.selectbox("Bathrooms", options=[1, 2, 3, 4, 5], index=1)
floors = st.sidebar.selectbox("Number of Floors", options=[1, 2, 3], index=0)
parking = st.sidebar.slider("Parking Spaces", min_value=0, max_value=3, value=1)
location_rating = st.sidebar.select_slider("Location Rating", options=[1, 2, 3, 4, 5], value=3)
property_age = st.sidebar.slider("Age of Property (Years)", min_value=0, max_value=50, value=5)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📋 Input Property Summary")
    input_data = {
        "Area_SqFt": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Floors": floors,
        "Parking": parking,
        "Location_Rating": location_rating,
        "Property_Age": property_age,
    }
    input_df = pd.DataFrame([input_data])
    
    display_df = pd.DataFrame({
        "Specification": ["Built-up Area", "Bedrooms", "Bathrooms", "Floors", "Parking Capacity", "Location Tier", "Property Age"],
        "Input Value": [f"{area} sq.ft", f"{bedrooms} BHK", f"{bathrooms} Baths", f"{floors} Floor(s)", f"{parking} Bay(s)", f"Tier {location_rating}/5", f"{property_age} Years"]
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

with col2:
    st.subheader("💰 Price Prediction")
    if st.button("Calculate Valuation", type="primary", use_container_width=True):
        prediction = model.predict(input_df)[0]
        st.success(f"### Estimated Price: ₹ {prediction:.2f} Lakhs")
        st.metric(label="Approximate Value (INR)", value=f"₹ {prediction * 100000:,.0f}")
        st.info(f"**Price per sq.ft:** ₹ {(prediction * 100000) / area:,.2f} / sq.ft")
    else:
        st.write("Click **Calculate Valuation** above to run the regression model.")

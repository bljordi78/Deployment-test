import streamlit as st
import pandas as pd
import joblib

from preprocessing.cleaning_data import preprocess
from predict.prediction import predict_with_error_handling

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("House Price Prediction App")

st.markdown("Please fill in the details below to receive the property price prediction:")

# Required Fields 
area = st.number_input("Area (m²)", min_value=1)
property_type = st.selectbox("Property Type", ["HOUSE", "APARTMENT"])
rooms_number = st.number_input("Number of Rooms", min_value=1)
zip_code = st.number_input("Zip Code", min_value=1000, max_value=9999)

# Optional Fields
lift = st.checkbox("Lift")
garden = st.checkbox("Garden")
swimming_pool = st.checkbox("Swimming Pool")
terrace = st.checkbox("Terrace")
parking = st.checkbox("Parking")

epc_score = st.selectbox("EPC Score (Energy Label)", ["", "A++", "A+", "A", "B", "C", "D", "E", "F", "G"])
epc_score = epc_score if epc_score else None

building_state = st.selectbox("Building Condition", ["", "AS_NEW", "JUST_RENOVATED", "GOOD", "TO_BE_DONE_UP", "TO_RENOVATE", "TO_RESTORE"])
building_state = building_state if building_state else None


# Button for prediction
if st.button("Predict Price"):
    try:
        input_dict = {
            "area": area,
            "property-type": property_type,
            "rooms-number": rooms_number,
            "zip-code": zip_code,
            "lift": lift,
            "garden": garden,
            "swimming-pool": swimming_pool,
            "terrace": terrace,
            "parking": parking,
            "epc-score": epc_score,
            "building-state": building_state
        }

        # Preprocess JSON user input
        preprocessed_data, preprocess_error = preprocess(input_dict)

        if preprocess_error:
            st.error(f"Preprocessing error: {preprocess_error}")
        else:
            prediction, prediction_error = predict_with_error_handling(preprocessed_data)

            label = "house" if property_type == "HOUSE" else "apartment"
            if prediction is not None:
                st.success(f"The predicted price for this {label} is € {prediction:,.0f}")
            else:
                st.error(f"Prediction failed: {prediction_error}")

    except Exception as e:
        st.error(f"Processing error: {e}")

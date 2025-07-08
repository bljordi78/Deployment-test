"""
TEST PRIOR TO APP
This test that preprocess and predict

How do we get a negative prediction?   ⚠️ Prediction is negative number:328009.1017728999. Return absolute number

"""

def test_end_to_end_prediction():
    # Step 1: Define your input (simulate API payload)
    sample_input = {
        "area": 95,
        "property-type": "APARTMENT",
        "rooms-number": 2,
        "zip-code": 1180,
        "lift": True,
        "garden": False,
        "swimming-pool": False,
        "terrace": True,
        "parking": True,
        "epc-score": "C",
        "building-state": "GOOD"
    }

    # Step 2: Run preprocessing
    preprocessed_data, error = preprocess(sample_input)

    if error:
        print(f"❌ Preprocessing failed: {error}")
        return

    print("✅ Preprocessing successful. Data:")
    print(preprocessed_data)

    # ✅ INSERT YOUR PREDICTION BLOCK HERE
    print("\n--- Running Prediction ---")
    predicted_price, prediction_error = predict_with_error_handling(preprocessed_data)

    if prediction_error:
        print(f"❌ Prediction failed: {prediction_error}")
    else:
        print(f"💶 Predicted house price: €{predicted_price:,.2f}")


from preprocessing.cleaning_data import preprocess
from predict.prediction import predict_with_error_handling
if __name__ == "__main__":
    test_end_to_end_prediction()

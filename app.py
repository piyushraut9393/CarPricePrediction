# Import necessary libraries
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"


def main() -> None:
    # Set page configuration
    st.set_page_config(
        page_title="Ford Car Price Prediction", page_icon="🚗", layout="wide"
    )

    st.title("Ford Car Price Prediction")
    st.write(
        "This application predicts the price of a Ford car based on its features. "
        "Please enter the details below to get the predicted price."
    )

    # --- User Inputs ---
    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input(
            "Year of Manufacture",
            min_value=1990,
            max_value=2026,
            value=2020,
            step=1,
        )
        mileage = st.number_input(
            "Mileage (in miles)",
            min_value=0,
            max_value=300000,
            value=20000,
            step=1000,
        )
        tax = st.number_input(
            "Tax (in £ / RS)", min_value=0.0, max_value=500.0, value=50.0, step=10.0
        )
        mpg = st.number_input(
            "Miles per Gallon (MPG)",
            min_value=0.0,
            max_value=150.0,
            value=30.0,
            step=1.0,
        )

    with col2:
        engine_size = st.number_input(
            "Engine Size (in liters)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
        )
        transmission = st.selectbox(
            "Transmission Type", ["Manual", "Automatic", "Semi-Auto"]
        )
        fuel_type = st.selectbox(
            "Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric", "Other"]
        )
        car_model = st.text_input(
            "Car Model (e.g. Fiesta, Focus, Kuga)", placeholder="Enter model name..."
        )

    predict_button = st.button("Predict Price")

    # --- Prediction Logic ---
    if predict_button:
        if not car_model.strip():
            st.warning("Please enter a car model before predicting.")
        else:
            try:
                # 1. Load trained artifacts
                model = joblib.load(MODEL_DIR / "model.pkl")
                scaler = joblib.load(MODEL_DIR / "scaler.pkl")
                encoded_columns = joblib.load(MODEL_DIR / "columns.pkl")

                # 2. Build initial DataFrame matching training features
                raw_input = pd.DataFrame(
                    {
                        "year": [year],
                        "mileage": [mileage],
                        "tax": [tax],
                        "mpg": [mpg],
                        "engineSize": [engine_size],
                        "transmission": [transmission],
                        "fuelType": [fuel_type],
                        "model": [car_model.strip()],
                    }
                )

                # 3. Apply One-Hot Encoding to categorical variables
                input_encoded = pd.get_dummies(raw_input)

                # 4. Align with model's expected features/columns
                input_data = input_encoded.reindex(
                    columns=encoded_columns, fill_value=0
                )

                # 5. Scale the full aligned feature matrix
                input_data = pd.DataFrame(
                    scaler.transform(input_data),
                    columns=encoded_columns,
                    index=input_data.index,
                )

                # 6. Predict price
                predicted_price = model.predict(input_data)[0]

                # 7. Display result
                st.success(f"**Predicted Price:** £{predicted_price:,.2f}")

            except FileNotFoundError:
                st.error(
                    "Model files not found! Ensure model.pkl, scaler.pkl, and columns.pkl exist in the model directory."
                )
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")


if __name__ == "__main__":
    main()
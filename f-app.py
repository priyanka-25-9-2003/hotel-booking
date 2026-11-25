
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import json
from pathlib import Path

app = Flask(__name__)

# Load the trained ML model
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "hotel_cancellation_model.pkl"

model = joblib.load(MODEL_PATH)

# All 29 features used during ML model training
FEATURES = [
    "hotel",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "assigned_room_type",
    "booking_changes",
    "deposit_type",
    "agent",
    "days_in_waiting_list",
    "customer_type",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "city"
]

# Numerical features
NUMERIC_FEATURES = [
    "lead_time",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "agent",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests"
]


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get user input
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        # Handle JSON entered inside a form field
        features = data.get("features", data)

        if isinstance(features, str):
            features = json.loads(features)

        # Check whether all required features are present
        missing_columns = [
            col for col in FEATURES if col not in features
        ]

        if missing_columns:
            raise ValueError(
                "Missing input columns: " + ", ".join(missing_columns)
            )

        # Arrange features in the same order as training
        input_data = pd.DataFrame(
            [{col: features[col] for col in FEATURES}]
        )

        # Convert numerical values into numeric format
        for col in NUMERIC_FEATURES:
            input_data[col] = pd.to_numeric(
                input_data[col], errors="coerce"
            )

        # Predict using the trained pipeline
        prediction = model.predict(input_data)[0]

        # Prediction probabilities
        probabilities = model.predict_proba(input_data)[0]

        canceled_probability = float(probabilities[1]) * 100
        not_canceled_probability = float(probabilities[0]) * 100

        if prediction == 1:
            result = "Booking May Be Cancelled"
            confidence = canceled_probability
        else:
            result = "Booking May Not Be Cancelled"
            confidence = not_canceled_probability

        # JSON response
        if request.is_json:
            return jsonify({
                "prediction": result,
                "predicted_class": int(prediction),
                "confidence": round(confidence, 2),
                "canceled_probability": round(
                    canceled_probability, 2
                ),
                "not_canceled_probability": round(
                    not_canceled_probability, 2
                )
            })

        # HTML response
        return render_template(
            "index.html",
            prediction=result,
            confidence=round(confidence, 2),
            canceled_probability=round(
                canceled_probability, 2
            ),
            not_canceled_probability=round(
                not_canceled_probability, 2
            )
        )

    except Exception as e:

        if request.is_json:
            return jsonify({"error": str(e)}), 400

        return render_template(
            "index.html",
            error=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True, port=5000)

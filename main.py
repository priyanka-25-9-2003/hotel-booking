from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from pathlib import Path
import pandas as pd
import joblib
import os
from fastapi.middleware.cors import CORSMiddleware
# Initialize FastAPI
app=FastAPI(
     title="Hotel Booking Cancellation Prediction",
    description="Machine Learning Classification API",
    version="1.0"
)
BASE_DIR = Path(__file__).resolve().parent
# Load trained ML model
#model = joblib.load("hotel_cancellation_model.pkl")
model = joblib.load(BASE_DIR / "hotel_cancellation_model.pkl")

# ⭐ Enable CORS
#app=FastAPI(title="Hotel Booking Cancellation Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500",
                   "http://127.0.0.1:5500"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Define input data
class HotelBooking(BaseModel):

    hotel: str
    lead_time: int
    arrival_date_year: int
    arrival_date_month: str
    arrival_date_week_number: int
    arrival_date_day_of_month: int
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    adults: int
    children: float
    babies: int
    meal: str
    country: str
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    reserved_room_type: str
    assigned_room_type: str
    booking_changes: int
    deposit_type: str
    agent: float
    company: float
    days_in_waiting_list: int
    customer_type: str
    adr: float
    required_car_parking_spaces: int
    total_of_special_requests: int
    city: str

# Home API
@app.get("/", response_class=HTMLResponse)
def home():
    html_file = BASE_DIR / "frontend_part.html"
    return html_file.read_text(encoding="utf-8")
   
@app.post("/predict")
def predict(data: HotelBooking):
    # Convert input data to Dataame
    input_data = pd.DataFrame([data.model_dump()])

    # Make prediction
    prediction = model.predict(input_data)[0]

    return {
        "prediction": int(prediction),
        "result": "Cancelled" if prediction == 1 else "Not Cancelled"
    }






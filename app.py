from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib  # For saving and loading models
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define the FastAPI app
app = FastAPI()

# Define the input data model for diabetes prediction
class DiabetesInput(BaseModel):
    age: float
    hypertension: int
    heart_disease: int
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float
    gender_Female: int
    gender_Male: int
    gender_Other: int
    smoking_history_current: int
    smoking_history_ever: int
    smoking_history_former: int
    smoking_history_never: int
    smoking_history_not_current: int

# Load the pre-trained Random Forest model
try:
    model = joblib.load('rf_model.joblib')
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    model = None  # Set to None if loading fails

# Create a root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Diabetes Prediction API! Use the /predict/ endpoint to make predictions."}

# Create a prediction endpoint
@app.post("/predict/")
def predict_diabetes(input_data: DiabetesInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check the logs.")

    # Convert the input data to a DataFrame for the model
    try:
        data = pd.DataFrame([input_data.dict().values()], columns=input_data.dict().keys())
        
        # Make a prediction
        prediction = model.predict(data)
        probability = model.predict_proba(data)[:, 1]  # Probability of being diabetic
        
        # Return the prediction result
        return {
            "prediction": int(prediction[0]),
            "probability_of_diabetes": float(probability[0])
        }
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
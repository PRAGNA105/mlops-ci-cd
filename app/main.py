from fastapi import FastAPI # imports the FastAPI class to create a web application
from pydantic import BaseModel # imports BaseModel from Pydantic to define the structure of the input data
import joblib # imports joblib to load the trained machine learning model
import os

# Create FastAPI app
app = FastAPI()

# Load trained model 
model_path = os.path.join(os.path.dirname(__file__), "model.pkl") 
model = joblib.load(model_path) # loads the trained model from the file model.pkl located in the same directory as this script

# Define input format
class InputData(BaseModel): # tells FastAPI that input JSON must match this structure
    features: list[float]

# Home route
@app.get("/")  # defines a route for the root URL ("/") that responds to GET requests
def home(): # When you open /, it shows the API is working.
    return {"message": "FastAPI ML model is running"}

# Prediction route
@app.post("/predict")
def predict(data: InputData): # When you send a POST request to /predict with JSON data, it will use the model to make a prediction based on the input features.
    prediction = model.predict([data.features])[0]
    return {"prediction": int(prediction)}
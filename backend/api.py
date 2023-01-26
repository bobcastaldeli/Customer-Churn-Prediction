"""This module contains the API endpoints for the models in the app."""

import pickle
import uvicorn
import pandas as pd
from fastapi import FastAPI, HTTPException
from churn_data import ChurnData


app = FastAPI()


@app.get("/")
def root():
    return {"message": "Welcome to the churn prediction APP!"}


@app.post("/predict")
def predict(data: ChurnData):
    """This endpoint takes in a ChurnData object and returns a prediction."""
    # Load the model
    with open("../models/model.pkl", "rb") as f:
        model = pickle.load(f)

    # transform the data into a dataframe
    new_customer = pd.DataFrame(data.dict(), index=[0])

    prediction = model.predict(new_customer)
    probability = model.predict_proba(new_customer)

    # Return the predictions as json
    prediction = prediction[0].item()
    probability = probability[0][1].item()

    churn_customer = {
        "customerID": data.customerID,
        "churn_prediction": prediction,
        "churn_probability": probability,
    }

    return churn_customer


if __name__ == "__main__":
    uvicorn.run(
        app, host="127.0.0.1", port=8000, log_level="info", reload=True
    )

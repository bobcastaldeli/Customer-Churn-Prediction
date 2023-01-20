"""This module contains the API endpoints for the models in the app."""

import pickle
import uvicorn
from fastapi import FastAPI, HTTPException
from churn_data import ChurnData


app = FastAPI()


@app.get("/")
def root():
    return {"message": "Welcome to the churn prediction APP!"}


@app.post("/predict")
def predict(data: ChurnData):
    """This endpoint takes in a ChurnData object and returns a prediction."""
    try:
        # Load the model
        model = pickle.load(open("models/model.pkl", "rb"))
        # Make prediction
        prediction = model.predict(data.dict())
        # Return prediction
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        app, host="127.0.0.1", port=8000, log_level="info", reload=True
    )

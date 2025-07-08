from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal

from preprocessing.cleaning_data import preprocess
from predict.prediction import predict_with_error_handling

app = FastAPI(title="Property Price Prediction API")

class HouseInput(BaseModel):

    area: int = Field(..., gt=0, description="Area in square meters (must be greater than 0)")
    property_type: Literal["HOUSE", "APARTMENT"] = Field(..., description="Type of property", alias="property-type")
    rooms_number: int = Field(..., gt=0, description="Number of rooms (must be greater than 0)", alias="rooms-number")
    zip_code: int = Field(..., ge=1000, le=9999, description="Belgian zip code (must have 4 digits)", alias="zip-code")

    # Optional fields
    lift: Optional[bool] = Field(None, description="Does the property have a lift?")
    garden: Optional[bool] = Field(None, description="Does the property have a garden?")
    swimming_pool: Optional[bool] = Field(None, description="Does the property have a Swimming pool?", alias="swimming-pool")
    terrace: Optional[bool] = Field(None, description="Does the property have a Terrace?")
    parking: Optional[bool] = Field(None, description="Does the property have a Parking?")
    epc_score: Optional[Literal["A++", "A+", "A", "B", "C", "D", "E", "F", "G"]] = Field(None, description="What is the EPC energy score?", alias="epc-score")
    building_state: Optional[Literal["AS_NEW", "JUST_RENOVATED", "GOOD", "TO_BE_DONE_UP", "TO_RENOVATE", "TO_RESTORE"]] = Field(None, description="What is the condition of the building?", alias="building-state")

class PredictionRequest(BaseModel):
    data: HouseInput


@app.get("/")
def health_check():
    return {"message": "alive"}

@app.get("/predict")
def explain_prediction_format():
     return {
        "message": "POST to /predict with JSON data matching the schema. See /docs for full OpenAPI specification.",
        "example": {
            "data": {
                "area": 100,
                "property-type": "HOUSE",
                "rooms-number": 3,
                "zip-code": 1050,
                "lift": True,
                "garden": False,
                "swimming-pool": False,
                "terrace": True,
                "parking": True,
                "epc-score": "C",
                "building-state": "GOOD"
            }
        }
    }

@app.post("/predict")
def predict_route(data: HouseInput): 
    try:
        # Get the data from JSON input 
        input_dict = data.dict(by_alias=True)

        # Preprocess JSON input
        preprocessed_data, preprocess_error = preprocess(input_dict)
        if preprocess_error:
            raise HTTPException(status_code=400, detail=preprocess_error)

        # Predict with preprocessed data
        prediction, prediction_error = predict_with_error_handling(preprocessed_data)
        if prediction_error:
            raise HTTPException(status_code=500, detail=prediction_error)

        return {
            "prediction": round(prediction, 2),
            "currency": "€",
            "message": f"Estimated property price: € {round(prediction):,}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
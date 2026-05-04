from fastapi import FastAPI, HTTPException
from schemas import (
    HousePredictionRequest,
    PredictionResponse,
    ModelInfoResponse,
    HealthCheckResponse
)
from api import (
    load_model_and_metadata,
    make_prediction,
    get_model_info,
    check_health
)

# Create FastAPI application
app = FastAPI(
    title="House Price Prediction API",
    description="Machine learning service for predicting US apartment prices based on 13 features",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Load model and metadata when the service starts"""
    success = load_model_and_metadata()
    if not success:
        print("WARNING: Failed to load model at startup")

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Check if the service is healthy and model is loaded.

    Returns:
        HealthCheckResponse with current service status
    """
    health_status = check_health()
    return HealthCheckResponse(
        status=health_status["status"],
        model_loaded=health_status["model_loaded"],
        metadata_loaded=health_status["metadata_loaded"],
        message=health_status["message"]
    )

@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """
    Get information about the loaded model.

    Returns:
        ModelInfoResponse with model metadata

    Raises:
        HTTPException: If model metadata is not loaded
    """
    try:
        # Get model info and return it
        info = get_model_info()
        return ModelInfoResponse(
            model_type=info["model_type"],
            version=info["version"],
            features=info["features"],
            training_date=info["training_date"],
            rmse=info["rmse"],
            description=info["description"]
        )
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: HousePredictionRequest):
    """
    Predict house price based on property features.

    Args:
        request: HousePredictionRequest with all 13 property features

    Returns:
        PredictionResponse with predicted price

    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    try:
        # Support both Pydantic v1 (.dict) and v2 (.model_dump)
        features_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()

        predicted_price = make_prediction(features_dict)

        model_version = get_model_info()["version"]

        return PredictionResponse(
            predicted_price=predicted_price,
            currency="USD",
            model_version=model_version
        )

    except ValueError as e:
        # Model not loaded error
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        # Other errors
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Root endpoint for basic information
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "House Price Prediction API",
        "version": "1.0.0",
        "endpoints": [
            "/health - Check service health",
            "/model/info - Get model information",
            "/predict - Make price prediction",
            "/docs - Interactive API documentation"
        ]
    }

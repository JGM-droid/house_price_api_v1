import joblib
import numpy as np
import json
import warnings
from typing import Dict, Any
from pathlib import Path
from sklearn.exceptions import InconsistentVersionWarning

# Global variable to store the loaded model
model = None  # Load the model at startup
metadata = None # Load the metadata at startup
BASE_DIR = Path(__file__).resolve().parent

# The serialized model was trained in an earlier sklearn version.
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

def load_model_and_metadata():
    """
    Load the trained model and metadata from disk.
    This function should:
    1. Load model.pkl using joblib.load()
    2. Load model_metadata.json using json.load()
    3. Store them in the global variables
    4. Return True if successful, False if there's an error
    """
    global model, metadata

    try:
        model_path = BASE_DIR / "model.pkl"
        metadata_path = BASE_DIR / "model_metadata.json"

        model = joblib.load(model_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        print("Model and metadata loaded successfully!")
        return True

    except Exception as e:
        print(f"Error loading model or metadata: {e}")
        return False

def make_prediction(house_features: Dict[str, Any]) -> float:
    """
    Make a price prediction for a single house.

    Args:
        house_features: Dictionary containing all 13 features

    Returns:
        Predicted price as a float

    This function should:
    1. Extract the features in the correct order (same order as training)
    2. Convert to numpy array with shape (1, 13)
    3. Use model.predict() to get prediction
    4. Return the prediction as a float
    """
    global model

    if model is None:
        raise ValueError("Model not loaded")

    if metadata is None:
        raise ValueError("Model metadata not loaded")

    feature_values = [house_features[feature] for feature in metadata["features"]]

    X = np.array([feature_values])

    prediction = model.predict(X)[0]

    # Round to 2 decimal places for currency
    return round(float(prediction), 2)

def get_model_info() -> Dict[str, Any]:
    """
    Get information about the loaded model.

    Returns:
        Dictionary containing model metadata

    This function should simply return the loaded metadata dictionary.
    If metadata is not loaded, it should raise an error.
    """
    global metadata

    if metadata is None:
        raise ValueError("Model metadata not loaded")

    return metadata

def check_health() -> Dict[str, Any]:
    """
    Check the health status of the service.

    Returns:
        Dictionary with health status information

    This function should:
    1. Check if model is loaded (model is not None)
    2. Check if metadata is loaded (metadata is not None)
    3. Return a dictionary with status information
    """
    global model, metadata

    model_loaded = model is not None
    metadata_loaded = metadata is not None
    is_healthy = model_loaded and metadata_loaded

    if is_healthy:
        message = "Service is healthy and ready to make predictions"
    elif not model_loaded and not metadata_loaded:
        message = "Service is unhealthy: Model and metadata not loaded"
    elif not model_loaded:
        message = "Service is unhealthy: Model not loaded"
    else:
        message = "Service is unhealthy: Metadata not loaded"

    health_status = {
        "status": "healthy" if is_healthy else "unhealthy",
        "model_loaded": model_loaded,
        "metadata_loaded": metadata_loaded,
        "message": message
    }

    return health_status

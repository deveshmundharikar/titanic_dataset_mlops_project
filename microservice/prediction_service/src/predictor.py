


#prediction_service/src/predictor.py
import joblib
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Required function for model compatibility - must match training_service implementation
def _preserve_age_column(X):
    """Preserve 'age' column name after SimpleImputer transformation."""
    if isinstance(X, pd.DataFrame):
        if X.shape[1] == 1 and X.columns[0] != 'age':
            X.columns = ['age']
    return X

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.pkl"
# Calculate log path relative to workspace root (go up from microservice/prediction_service/src to workspace root)
LOG_PATH = Path(__file__).resolve().parent.parent.parent.parent / ".cursor" / "debug.log"

# Add training_service to path for model compatibility
import sys
MICROSERVICE_PATH = Path(__file__).resolve().parent.parent.parent
if str(MICROSERVICE_PATH) not in sys.path:
    sys.path.append(str(MICROSERVICE_PATH))

# #region agent log
def _log(hypothesis_id, message, data):
    try:
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": hypothesis_id,
                "location": "predictor.py",
                "message": message,
                "data": data,
                "timestamp": int(datetime.now().timestamp() * 1000)
            }) + "\n")
    except:
        pass
# #endregion agent log

# Lazy loading: Load model on first use to ensure we get the latest version
_model = None
_preprocessor = None
_EXPECTED_COLUMNS = None

def _load_model(force_reload=False):
    """Load model and extract preprocessor info. Called lazily on first use."""
    global _model, _preprocessor, _EXPECTED_COLUMNS
    if _model is None or force_reload:
        _model = joblib.load(MODEL_PATH)
        _preprocessor = _model.named_steps['preprocessor']
        
        # #region agent log
        _log("A", "Model loaded", {
            "has_feature_names_in_": hasattr(_preprocessor, 'feature_names_in_'),
            "preprocessor_type": type(_preprocessor).__name__,
            "transformers": list(_preprocessor.named_transformers_.keys()) if hasattr(_preprocessor, 'named_transformers_') else None
        })
        # #endregion agent log
        
        # Check age pipeline structure
        # #region agent log
        try:
            age_pipe = _preprocessor.named_transformers_['age']
            winsorizer = age_pipe.named_steps['outliers']
            has_preserve = 'preserve_names' in age_pipe.named_steps
            _log("E", "Winsorizer inspection", {
                "has_variables_": hasattr(winsorizer, 'variables_'),
                "variables_": list(winsorizer.variables_) if hasattr(winsorizer, 'variables_') else None,
                "winsorizer_type": type(winsorizer).__name__,
                "has_preserve_names_step": has_preserve,
                "age_pipe_steps": list(age_pipe.named_steps.keys())
            })
        except Exception as e:
            _log("E", "Winsorizer inspection failed", {"error": str(e)})
        # #endregion agent log
        
        if hasattr(_preprocessor, 'feature_names_in_'):
            _EXPECTED_COLUMNS = list(_preprocessor.feature_names_in_)
            # #region agent log
            _log("A", "Using feature_names_in_", {"expected_columns": _EXPECTED_COLUMNS})
            # #endregion agent log
        else:
            _EXPECTED_COLUMNS = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'deck']
            # #region agent log
            _log("A", "Using fallback columns", {"expected_columns": _EXPECTED_COLUMNS})
            # #endregion agent log
    return _model, _preprocessor, _EXPECTED_COLUMNS

# Model will be loaded lazily on first predict() call


def predict(data: dict) -> dict:
    """
    Generate a prediction for the given input data.
    
    Args:
        data: Dictionary with passenger features
        
    Returns:
        Dictionary with prediction and probability
    """
    # Reload model to ensure we have the latest version (force reload for debugging)
    model, preprocessor, expected_columns = _load_model(force_reload=True)
    
    # #region agent log
    _log("B", "predict function entry", {"input_keys": list(data.keys()), "input_values": {k: str(v) for k, v in data.items()}})
    # #endregion agent log
    
    # Create DataFrame from input data
    df = pd.DataFrame([data])
    
    # #region agent log
    _log("B", "DataFrame created", {"df_columns": list(df.columns), "df_shape": df.shape, "df_dtypes": {k: str(v) for k, v in df.dtypes.items()}})
    # #endregion agent log
    
    # Ensure all required columns are present
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        # #region agent log
        _log("C", "Missing columns error", {"missing_cols": list(missing_cols), "df_columns": list(df.columns), "expected_columns": expected_columns})
        # #endregion agent log
        raise ValueError(f"Missing required columns: {missing_cols}. Received columns: {list(df.columns)}")
    
    # Reorder columns to match the exact order used during training
    # This is critical for ColumnTransformer with remainder='passthrough'
    df = df[expected_columns]
    
    # #region agent log
    _log("B", "DataFrame reordered", {"df_columns_after": list(df.columns), "df_shape": df.shape, "expected_order": expected_columns})
    # #endregion agent log
    
    # #region agent log
    _log("D", "Before model.predict", {"df_columns": list(df.columns), "df_values": df.iloc[0].to_dict()})
    # Check if model has preserve_names step
    try:
        age_pipe = preprocessor.named_transformers_['age']
        has_preserve = 'preserve_names' in age_pipe.named_steps
        _log("D", "Model structure check", {"has_preserve_names_step": has_preserve, "age_pipe_steps": list(age_pipe.named_steps.keys())})
    except Exception as e:
        _log("D", "Model structure check failed", {"error": str(e)})
    # #endregion agent log
    
    try:
        prediction = model.predict(df)[0]
        # #region agent log
        _log("D", "After model.predict", {"prediction": int(prediction)})
        # #endregion agent log
    except Exception as e:
        # #region agent log
        _log("F", "model.predict error", {"error_type": type(e).__name__, "error_message": str(e), "df_columns": list(df.columns)})
        # #endregion agent log
        raise
    
    result = {"prediction": int(prediction)}
    
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(df)[0]
        # Assumes binary classification and returns the probability of the positive class
        result["probability"] = float(probabilities[1])
        
    return result


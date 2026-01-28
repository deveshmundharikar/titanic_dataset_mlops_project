






#prediction_service/src/predictor.py
import joblib
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Add training_service to path for model compatibility
import sys
MICROSERVICE_PATH = Path(__file__).resolve().parent.parent.parent
TRAINING_SERVICE_PATH = MICROSERVICE_PATH / "training_service"
TRAINING_SERVICE_SRC_PATH = TRAINING_SERVICE_PATH / "src"

# Add all necessary paths
paths_to_add = [
    str(MICROSERVICE_PATH),
    str(TRAINING_SERVICE_PATH), 
    str(TRAINING_SERVICE_SRC_PATH)
]

for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import all necessary modules that the model might need
try:
    # Import training service modules with multiple import strategies
    try:
        from training_service.src.preprocess import _preserve_age_column
    except ImportError:
        try:
            from src.preprocess import _preserve_age_column
        except ImportError:
            try:
                import preprocess
                _preserve_age_column = preprocess._preserve_age_column
            except ImportError:
                # Define fallback function
                def _preserve_age_column(X):
                    """Preserve 'age' column name after SimpleImputer transformation."""
                    if isinstance(X, pd.DataFrame):
                        if X.shape[1] == 1 and X.columns[0] != 'age':
                            X.columns = ['age']
                    return X
    
    # Import sklearn modules that might be needed
    from sklearn.preprocessing import FunctionTransformer
    from feature_engine.encoding import CountFrequencyEncoder
    from feature_engine.outliers import Winsorizer
    from feature_engine.encoding import OrdinalEncoder
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    # Make the function available in multiple namespaces for pickle compatibility
    import sys
    current_module = sys.modules[__name__]
    setattr(current_module, '_preserve_age_column', _preserve_age_column)
    
    # Also add to sys.modules for pickle to find
    if 'src' not in sys.modules:
        import types
        src_module = types.ModuleType('src')
        sys.modules['src'] = src_module
    if 'src.preprocess' not in sys.modules:
        import types
        preprocess_module = types.ModuleType('src.preprocess')
        preprocess_module._preserve_age_column = _preserve_age_column
        sys.modules['src.preprocess'] = preprocess_module
        
except ImportError as e:
    print(f"Warning: Could not import training modules: {e}")
    # Define fallback function
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
        
        # Fix n_jobs issue that causes column access problems
        if hasattr(_preprocessor, 'n_jobs'):
            _preprocessor.n_jobs = 1
        
        # Also fix n_jobs in any nested transformers
        if hasattr(_preprocessor, 'named_transformers_'):
            for name, transformer in _preprocessor.named_transformers_.items():
                if hasattr(transformer, 'n_jobs'):
                    transformer.n_jobs = 1
                # Check if it's a pipeline with steps
                if hasattr(transformer, 'named_steps'):
                    for step_name, step in transformer.named_steps.items():
                        if hasattr(step, 'n_jobs'):
                            step.n_jobs = 1
        
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
        
        # Use the exact columns the model was trained with
        if hasattr(_preprocessor, 'feature_names_in_'):
            _EXPECTED_COLUMNS = list(_preprocessor.feature_names_in_)
            # #region agent log
            _log("A", "Using feature_names_in_", {"expected_columns": _EXPECTED_COLUMNS})
            # #endregion agent log
        else:
            # Fallback: use the standard Titanic columns
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
    # Load model with force reload to get the fresh simple model
    model, preprocessor, expected_columns = _load_model(force_reload=True)
    
    # #region agent log
    _log("B", "predict function entry", {"input_keys": list(data.keys()), "input_values": {k: str(v) for k, v in data.items()}})
    # #endregion agent log
    
    # Create DataFrame from input data with all expected columns
    # Start with the expected columns and fill with defaults
    row_data = {}
    
    # Debug: print what columns the model expects
    print(f"Model expects columns: {expected_columns}")
    print(f"Received data keys: {list(data.keys())}")
    
    for col in expected_columns:
        if col in data:
            row_data[col] = data[col]
        else:
            # Set default values for missing columns
            if col == 'embarked':
                row_data[col] = 'S'
            elif col == 'deck':
                row_data[col] = 'Unknown'
            elif col in ['pclass', 'sibsp', 'parch']:
                row_data[col] = 0
            elif col in ['age', 'fare']:
                row_data[col] = 0.0
            elif col == 'sex':
                row_data[col] = 'male'
            else:
                row_data[col] = 0
    
    # Create DataFrame with exact columns in exact order
    df = pd.DataFrame([row_data])
    
    # Ensure column order matches exactly
    df = df[expected_columns]
    
    print(f"Final DataFrame columns: {list(df.columns)}")
    print(f"DataFrame values: {df.iloc[0].to_dict()}")
    
    # #region agent log
    _log("B", "DataFrame created", {"df_columns": list(df.columns), "df_shape": df.shape, "expected_columns": expected_columns})
    # #endregion agent log
    
    # #region agent log
    _log("D", "Before model.predict", {"df_columns": list(df.columns), "df_values": df.iloc[0].to_dict()})
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


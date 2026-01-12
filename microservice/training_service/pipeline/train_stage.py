import logging
import sys
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from training_service.src.preprocess import (
    load_data,
    data_preprocessing,
    split_features_target,
    split_data,
    imputation_pipeline,
    column_transformer,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_model_params():
    return {
        'bootstrap': True,
        'ccp_alpha': 0.0,
        'class_weight': None,
        'criterion': 'gini',
        'max_depth': 6,
        'max_features': 'sqrt',
        'max_leaf_nodes': None,
        'max_samples': 0.8,
        'min_impurity_decrease': 0.0,
        'min_samples_leaf': 1,
        'min_samples_split': 2,
        'min_weight_fraction_leaf': 0.0,
        'monotonic_cst': None,
        'n_estimators': 300,
        'n_jobs': -1,
        'oob_score': False,
        'random_state': 30,
        'verbose': 0,
        'warm_start': False
    }

def run_training_pipeline(data_path: str = None, model_save_path: str = None):
    """
    Executes the end-to-end training pipeline.
    """
    try:
        # 1. Load Data
        df = load_data(data_path)
        
        # 2. Preprocess Data
        data_preprocessing(df)
        
        # 3. Split Features and Target
        X, y = split_features_target(df)
        
        # 4. Split Data
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # 5. Get Preprocessor
        age_pipe, embarked_pipe, deck_pipe = imputation_pipeline()
        preprocessor = column_transformer(age_pipe, embarked_pipe, deck_pipe)

        # 6. Create Model Pipeline
        model_pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ("ml", RandomForestClassifier(**get_model_params()))
        ])
        
        # 7. Train Model
        logger.info("Starting model training...")
        model_pipe.fit(X_train, y_train)
        logger.info("Model trained successfully")
        
        # 8. Evaluate Model
        y_pred = model_pipe.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred)
        }
        
        logger.info("Model training completed")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        if model_save_path:
            # Create directory if it doesn't exist
            model_path_obj = Path(model_save_path)
            model_path_obj.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving model to {model_save_path}")
            joblib.dump(model_pipe, model_save_path)
            
        return model_pipe, metrics

    except Exception:
        logger.exception("Error during model training")
        raise

if __name__ == "__main__":
    # Use absolute path resolution for reliability
    data_file = PROJECT_ROOT / "notebook" / "data" / "titanic.csv"
    
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
    else:
        model_path = PROJECT_ROOT / "prediction_service" / "artifacts" / "model.pkl"
        run_training_pipeline(str(data_file), str(model_path))

#!/usr/bin/env python3
"""
Run training pipeline with DagsHub MLflow tracking - Simple version
"""
import os
import sys
from pathlib import Path
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set up DagsHub authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

# Initialize DagsHub
import dagshub
dagshub.init(repo_owner='deveshmundharikar', repo_name='titanic_dataset_mlops_project', mlflow=True)

# Import MLflow functions directly
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

def setup_mlflow(experiment_name: str = None):
    """Set MLflow tracking URI and experiment."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/deveshmundharikar/titanic_dataset_mlops_project")
    experiment_name = experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME", "titanic_experiment")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

def log_training_run(model, params: dict, metrics: dict, run_name: str = None, registered_model_name: str = None):
    """Logs training run to MLflow."""
    run_name = run_name or os.getenv("MLFLOW_RUN_NAME", "random_forest_run")
    registered_model_name = registered_model_name or os.getenv("MLFLOW_REGISTERED_MODEL", "TitanicSurvivalModel")
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=registered_model_name
        )
        print(f"Model logged and registered: {registered_model_name}")

# Import training functions
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Import preprocessing functions
from src.preprocess import (
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

def run_training_with_mlflow():
    """Run training pipeline with MLflow tracking"""
    try:
        # Setup MLflow
        logger.info("Setting up MLflow tracking...")
        setup_mlflow("titanic_experiment")
        logger.info("MLflow setup successful")
        
        # Data paths
        data_file = PROJECT_ROOT / "microservice" / "notebook" / "data" / "titanic.csv"
        model_path = PROJECT_ROOT / "microservice" / "prediction_service" / "artifacts" / "model.pkl"
        
        # 1. Load Data
        df = load_data(str(data_file))
        
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
        model_params = get_model_params()
        model_pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ("ml", RandomForestClassifier(**model_params))
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
        
        # 9. Log to MLflow
        logger.info("Logging to MLflow...")
        log_training_run(
            model=model_pipe,
            params=model_params,
            metrics=metrics,
            run_name="titanic_training_run",
            registered_model_name="TitanicSurvivalModel"
        )
        logger.info("MLflow logging successful")
        
        # 10. Save model
        model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving model to {model_path}")
        joblib.dump(model_pipe, model_path)
        
        return model_pipe, metrics
        
    except Exception as e:
        logger.exception(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    print("Running training pipeline with DagsHub MLflow tracking...")
    print(f"DagsHub URI: {os.getenv('MLFLOW_TRACKING_URI')}")
    print(f"Username: {os.getenv('MLFLOW_TRACKING_USERNAME')}")
    
    try:
        model, metrics = run_training_with_mlflow()
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY WITH MLFLOW!")
        print("="*50)
        print("Final Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print(f"\nCheck your DagsHub repo for MLflow results:")
        print(f"https://dagshub.com/deveshmundharikar/titanic_dataset_mlops_project")
        print("="*50)
        
    except Exception as e:
        print(f"Error during training: {e}")
        print("\nIf authentication failed, you may need to:")
        print("1. Get a new access token from: https://dagshub.com/user/settings/tokens")
        print("2. Update MLFLOW_TRACKING_PASSWORD in your .env file")
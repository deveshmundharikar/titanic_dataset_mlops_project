# Titanic Survival Prediction Microservice

A complete machine learning microservice for predicting Titanic passenger survival with separate training and prediction services.

## Project Structure

```
microservice/
├── notebook/data/titanic.csv          # Training dataset
├── prediction_service/                # Prediction API service
│   ├── api/main.py                   # FastAPI prediction endpoints
│   ├── src/
│   │   ├── predictor.py              # Core prediction logic
│   │   └── schema.py                 # Pydantic data models
│   └── artifacts/model.pkl           # Trained model file
├── training_service/                  # Training API service
│   ├── api/main.py                   # FastAPI training endpoints
│   ├── config/config.yml             # Training configuration
│   ├── pipeline/train_stage.py       # Training pipeline
│   └── src/
│       ├── data_loader.py            # Data loading utilities
│       └── preprocess.py             # Data preprocessing
├── requirements.txt                   # Python dependencies
└── .env                              # Environment variables
```

## Setup Instructions

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)

venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

```



# Start the training API server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload

# Open API documentation
Start-Process "http://localhost:8001/docs"
```

#### Run Training Pipeline Directly
```bash
# From project root
cd microservice

# Run training pipeline standalone
python training_service/pipeline/train_stage.py

# Or run with custom parameters
python -c "from training_service.pipeline.train_stage import run_training_pipeline; run_training_pipeline('notebook/data/titanic.csv', 'prediction_service/artifacts/model.pkl')"
```

#### Training API Endpoints
- **Health Check**: `GET http://localhost:8001/health`
- **Train Model**: `POST http://localhost:8001/train`
  ```json
  {
    "dataset_path": "notebook/data/titanic.csv",
    "model_name": "titanic_survival_model"
  }
  ```

### Prediction Service

#### Start Prediction API Server
```bash
# Navigate to prediction service directory
cd prediction_service

# Start the prediction API server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Open API documentation
Start-Process "http://localhost:8000/docs"
```

#### Prediction API Endpoints
- **Health Check**: `GET http://localhost:8000/health`
- **Make Prediction**: `POST http://localhost:8000/predict`
  ```json
  {
    "pclass": 3,
    "sex": "male",
    "age": 22,
    "sibsp": 1,
    "parch": 0,
    "fare": 7.25,
    "embarked": "S",
    "deck": "Unknown"
  }
  ```

### Data Processing Files

#### Data Loader
```bash
# Run data loading utilities
cd training_service/src
python data_loader.py
```

#### Preprocessing
```bash
# Run preprocessing pipeline
cd training_service/src
python preprocess.py
```

### Configuration Files

#### Training Configuration (`training_service/config/config.yml`)
- Modify model parameters, training settings, and API configuration
- No direct execution required - loaded by training service

#### Requirements Installation
```bash
# Install all dependencies
pip install -r requirements.txt

# Install specific packages
pip install pandas scikit-learn fastapi uvicorn
```


## Quick Start Commands

```bash
# 1. Setup environment
python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt




```

## File Execution Summary

| File | Purpose | Execution Command |
|------|---------|-------------------|
| `training_service/api/main.py` | Training API | `uvicorn api.main:app --port 8001` |
| `prediction_service/api/main.py` | Prediction API | `uvicorn api.main:app --port 8000` |
| `training_service/pipeline/train_stage.py` | Training Pipeline | `python train_stage.py` |
| `training_service/src/data_loader.py` | Data Loading | `python data_loader.py` |
| `training_service/src/preprocess.py` | Preprocessing | `python preprocess.py` |
| `prediction_service/src/predictor.py` | Prediction Logic | Imported by API |
| `prediction_service/src/schema.py` | Data Models | Imported by API |
| `training_service/config/config.yml` | Configuration | Loaded by services |
| `requirements.txt` | Dependencies | `pip install -r requirements.txt` |





## Mlflow :- command 


python microservice/training_service/run_dagshub_simple.py




# Method 2: Separate lines
cd microservice/training_service
python -m uvicorn api.main:app --port 8001 --reload




## dvc command
dvc push 


## Delete and pull
del microservice\notebook\data\titanic.csv
dvc pull

## Status
dvc status 

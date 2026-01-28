

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






  how to execute this file 
cd microservice/prediction_service

python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
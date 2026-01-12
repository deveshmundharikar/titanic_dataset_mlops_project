import pandas as pd 
import numpy as np 
import logging

from training_service.src.data_loader import load_data

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from feature_engine.encoding import CountFrequencyEncoder
from feature_engine.outliers import Winsorizer
from feature_engine.encoding import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn import set_config

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

set_config(transform_output='pandas')

if __name__ == "__main__":
    df=load_data()
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Size: {df.size}")

def data_preprocessing(df):
    # Drop unnecessary columns if they exist in the DataFrame
    cols_to_drop = ["adult_male", "alive", "alone", "who", "class", "embark_town"]
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    if existing_cols_to_drop:
        df.drop(columns=existing_cols_to_drop, axis=1, inplace=True)
        logger.info(f"Dropped columns: {existing_cols_to_drop}")
    else: 
        logger.info(f"No columns to drop from: {cols_to_drop}")

def split_features_target(df: pd.DataFrame, target='survived'):
    X = df.drop(columns=[target], axis=1)
    y = df[target]
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logger.info(f"Shape of X_train: {X_train.shape}")
    logger.info(f"Shape of X_test: {X_test.shape}")
    logger.info(f"Shape of y_train: {y_train.shape}")
    logger.info(f"Shape of y_test: {y_test.shape}")
    return X_train, X_test, y_train, y_test


def _preserve_age_column(X):
    """Preserve 'age' column name after SimpleImputer transformation."""
    if isinstance(X, pd.DataFrame):
        if X.shape[1] == 1 and X.columns[0] != 'age':
            X.columns = ['age']
    return X

def imputation_pipeline():
    # Create a transformer to preserve column names after SimpleImputer
    preserve_names = FunctionTransformer(
        func=_preserve_age_column,
        validate=False
    )
    
    age_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy='median')),
        ("preserve_names", preserve_names),
        ("outliers", Winsorizer(capping_method='gaussian', fold=3, variables=None)),
        ("scale", StandardScaler())
    ])
    embarked_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy='most_frequent')),
        ("count_encoder", CountFrequencyEncoder(encoding_method="count")),
        ("scale", MinMaxScaler())
    ])
    deck_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy='most_frequent')),
        ("ordi", OrdinalEncoder()),
        ("scale", MinMaxScaler())
    ])
    logger.info("Imputation pipelines created for: age, embarked, deck")
    return age_pipe, embarked_pipe, deck_pipe


def column_transformer(age_pipe, embarked_pipe, deck_pipe):
    preprocessor = ColumnTransformer(
        transformers=[
            ('age', age_pipe, ['age']),
            ('embark', embarked_pipe, ['embarked']),
            ('deck', deck_pipe, ['deck']),
            ('sex', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['sex'])
        ],
        remainder='passthrough',
        n_jobs=-1
    )
    logger.info("ColumnTransformer created with specified pipelines and transformers")
    return preprocessor

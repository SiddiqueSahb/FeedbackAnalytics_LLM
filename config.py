import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Data paths
    DATA_PATH:             str = "data/processed/reviews_unified.parquet"
    RAW_DATA_PATH:         str = "data/raw"

    # MLflow (free, local) 
    MLFLOW_TRACKING_URI:  str = "mlruns"      #This tells MLflow where to save experiment data.URI is simply the address or location that MLflow uses to read and write experiment information.
    MLFLOW_EXPERIMENT:    str = "Feedback_Sentiment"  #This is simply the name of your experiment


settings = Settings()


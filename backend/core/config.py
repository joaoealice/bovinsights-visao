from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    roboflow_api_key: str
    roboflow_model_id: str = "cattle-dataset-behavior-cqtzu"
    roboflow_model_version: int = 1
    roboflow_count_model_id: str = "cow-count"
    roboflow_count_model_version: int = 2
    roboflow_workspace: str = ""
    roboflow_project: str = ""
    api_env: str = "development"
    api_secret_key: str = "change_this_in_production"
    max_image_size_mb: int = 5
    inference_interval_seconds: int = 3
    allowed_origins: str = "http://localhost:3000"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()

import os
import shutil
import tempfile
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from sqlalchemy.engine import URL

load_dotenv()


db_url = os.getenv("DATABASE_URL_MLFLOW")


mlflow.set_tracking_uri(db_url)

MODEL_URI = "models:/Oil_Well_Fault_Detector@champion"

destination = Path("models/champion")
destination.parent.mkdir(parents=True, exist_ok=True)

with tempfile.TemporaryDirectory() as temp_dir:

    downloaded_path = mlflow.artifacts.download_artifacts(
        artifact_uri=MODEL_URI,
        dst_path=temp_dir
    )
    shutil.copytree(downloaded_path, destination)

print("Champion model exported to:", destination.resolve())
print("MLflow model file exists:", (destination / "MLmodel").is_file())
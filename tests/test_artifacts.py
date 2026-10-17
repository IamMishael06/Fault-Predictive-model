
from pathlib import Path

import pandas as pd
import mlflow.sklearn


# Locate the project root regardless of where pytest is executed
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_champion_model_loads():
    """Verify that the exported champion model can be loaded."""

    model_path = PROJECT_ROOT / "models" / "champion"

    assert model_path.exists(), "Champion model directory is missing"

    model = mlflow.sklearn.load_model(str(model_path))

    assert hasattr(model, "predict"), "Loaded model cannot predict"
    assert set(model.classes_) == {0, 101, 102, 105, 107, 108}


def test_synthetic_stream_data():
    """Verify that the synthetic sensor dataset is available and readable."""

    csv_path = (
        PROJECT_ROOT
        / "cleaned_csv"
        / "synthetic_oil_stream_1s_30_predictions.csv"
    )

    assert csv_path.exists(), "Synthetic CSV is missing"

    df = pd.read_csv(csv_path)

    assert not df.empty, "Synthetic CSV contains no observations"

    # Enough 1-second readings for the 120-minute history buffer
    assert len(df) >= 240 * 30

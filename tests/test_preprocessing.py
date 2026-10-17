
import numpy as np
import pandas as pd
import pytest

from api.preprocessing.preprocessing import (
    process_file,
    roll_win,
    preproccesing,
)


def test_process_file_resampling(tmp_path):
    """
    Check 1-second to 30-second resampling,
    class labels, and gas-lift handling.
    """
    timestamps = pd.date_range(
        "2026-01-01", periods=90, freq="s"
    )

    df = pd.DataFrame({
        "timestamp": timestamps,
        "P-TPT": [100.0] * 90,
        "T-TPT": [70.0] * 90,
        "P-MON-CKP": [40.0] * 90,
        "T-JUS-CKP": [25.0] * 90,
        "P-JUS-CKGL": [np.nan] * 30 + [20.0] * 60,
        "QGL": [np.nan] * 30 + [4.0] * 60,
        "P-PDG": [500.0] * 90,
        "Instance_id": [1] * 90,
        "class": [0] * 30 + [101] * 30 + [102] * 30,
    })

    csv_path = tmp_path / "test_sensor_data.csv"
    df.to_csv(csv_path, index=False)

    result = process_file(str(csv_path))

    # 90 seconds should produce three 30-second observations.
    assert len(result) == 3

    assert result["class"].tolist() == [0, 101, 102]

    assert result["gas_lift_active"].tolist() == [0, 1, 1]

    assert result["QGL"].tolist() == [0.0, 4.0, 4.0]

    assert "P-PDG" not in result.columns
    
    assert "Instance_id" not in result.columns

    assert (
        result.index[1] - result.index[0]
        == pd.Timedelta(seconds=30)
    )


def test_rolling_window_features():
    """
    Verify that rolling statistics and EWM
    are calculated correctly for constant sensor data.
    """

    timestamps = pd.date_range(
        "2026-01-01",
        periods=240,
        freq="30s",
        name="timestamp",
    )

    df = pd.DataFrame({
        "P-TPT": [100.0] * 240,
        "T-TPT": [70.0] * 240,
        "P-MON-CKP": [40.0] * 240,
        "T-JUS-CKP": [25.0] * 240,
        "P-JUS-CKGL": [20.0] * 240,
        "QGL": [4.0] * 240,
    }, index=timestamps)

    result = roll_win(df)

    last_row = result.iloc[-1]

    assert last_row["P-TPT_mean_10min"] == pytest.approx(100)
    assert last_row["P-TPT_std_10min"] == pytest.approx(0)

    assert last_row["P-TPT_mean_120min"] == pytest.approx(100)
    assert last_row["P-TPT_std_120min"] == pytest.approx(0)

    assert last_row["P-TPT_ewm"] == pytest.approx(100)

    # January's cyclical month encoding
    assert last_row["month_sin"] == pytest.approx(0.5)
    assert last_row["month_cos"] == pytest.approx(np.sqrt(3) / 2)


def test_engineered_features_and_stability():
    """
    Verify pressure-based features, sign changes,
    and stability streak resetting.
    """

    timestamps = pd.date_range(
        "2026-01-01",
        periods=5,
        freq="30s",
        name="timestamp",
    )

    df = pd.DataFrame({
        "P-TPT": [100, 110, 150, 100, 110],
        "T-TPT": [70] * 5,
        "P-MON-CKP": [40] * 5,
        "T-JUS-CKP": [25] * 5,
        "QGL": [4] * 5,
    }, index=timestamps)

    result = preproccesing(df)

    # Check derived features at the first observation.
    assert result["T_P_divergence"].iloc[0] == pytest.approx(70 / 101)
    assert result["choke_pressure_diff"].iloc[0] == -60
    assert result["productivity_index"].iloc[0] == 20
    assert result["hydrate_risk"].iloc[0] == 5

    # Pressure of 150 falls outside the stable range.
    assert result["all_stable"].tolist() == [1, 1, 0, 1, 1]

    # The stability streak must restart after instability.
    assert result["stability_streak"].tolist() == [0, 1, 0, 0, 1]

    # Changes in the direction of pressure movement.
    assert result["P_TPT_sign_changes"].tolist() == [0, 0, 0, 1, 2]

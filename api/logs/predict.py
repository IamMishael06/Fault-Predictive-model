from fastapi import FastAPI, Cookie, UploadFile, File, HTTPException, status, Depends, APIRouter
from sqlmodel import Field, Session, SQLModel, create_engine, select
from pydantic import BaseModel
from db.db import engine, Engineer, Log
from datetime import datetime
from auth.security import decode_access_token
import mlflow
import asyncio
from collections import deque
import csv
import os
from dotenv import load_dotenv
import mlflow.sklearn
from mlflow import MlflowClient
import json
from preprocessing.preprocessing import roll_win, preproccesing, process_file
import pandas as pd
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse
from pathlib import Path
import jwt
from auth.token_blacklist import token_blacklist



BASE_DIR = Path(__file__).resolve().parents[2]


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


class PredictionInput(SQLModel):
    timestamp : datetime = Field(default_factory=datetime.now)
    P_PDG : int  = Field(alias="P_PDG", description="Pressure at the pump discharge gauge")
    P_TPT : int = Field(alias="P_TPT", description="Pressure at the pump suction gauge")
    T_TPT : int = Field(alias="T_TPT", description="Temperature at the pump suction gauge")
    P_MON_CKP : int = Field(alias="P_MON_CKP", description="Pressure at the monitor checkpoint")
    T_JUS_CKP : int = Field(alias="T_JUS_CKP", description="Temperature at the justification checkpoint")
    P_JUS_CKGL : int = Field(alias="P_JUS_CKGL", description="Pressure at the justification checkpoint (glow)")
    QGL : int = Field(alias="QGL", description="Flow rate at the glow checkpoint")
    well_id : int 




def get_current_engineer(token: str = Depends(oauth2_scheme)):
    # Here you would decode the JWT token and retrieve the engineer's information (Here, it is the email)
    if token in token_blacklist:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked (Logged out). Please log in again.",
        )
    try:
        payload = decode_access_token(token)
        engineer_id = payload.get("sub")

        print(engineer_id)

        if engineer_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")

        engineer_id = int(engineer_id)

        with Session(engine) as session:
            statement = select(Engineer).where(Engineer.engineer_id == engineer_id)
            engineer = session.exec(statement).first()
            if not engineer:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
            
            return engineer
        
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")



# ---------------------------------------------------------
# LOAD MODEL FROM MLFLOW MODEL FOLDER CONTAINING SAVED ARTIFCATS
# ---------------------------------------------------------
MODEL_PATH = (Path(__file__).resolve().parents[2] / "models" / "champion")

model = mlflow.sklearn.load_model(str(MODEL_PATH))


router = APIRouter(prefix='/predict', tags=['Predictions'])

# ---------------------------------------------------------
# MODEL SIMULATION CODE 
# ---------------------------------------------------------

QUEUE_1S = deque() # Create an empty queue for the 1sec readings
QUEUE_30S = deque() # Create an empty queue for the 30sec readings

sensor_reading = []
async def readings_simulator(file_path : str, engineer_id: int, well_id: str):
    """
        Simulates live 1-second sensor readings.

        Every 30 raw readings are resampled into one
        30-second observation.

        Once 240 30-second observations exist,
        feature engineering is performed and a
        prediction is made on the latest observation.
    """

    QUEUE_1S.clear()
    QUEUE_30S.clear()

    with open(file_path, mode='r', encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        print(f"Starting Stream...")

        for row in reader:

             # -----------------------------------------
            # 1. RECEIVE ONE 1-SECOND SENSOR READING
            # -----------------------------------------
            QUEUE_1S.append(row)
            await asyncio.sleep(0.01) # 0.01 for debugging purposes

            if len(QUEUE_1S) >= 30:
                print("30 readings collected. Creating 30-second observation.")


                snapshot_path = BASE_DIR / "cleaned_csv"/ "output_snapshot.csv"

                with open(snapshot_path, mode='w', newline='', encoding='utf-8') as out_file:
                    writer = csv.DictWriter(out_file, fieldnames=reader.fieldnames)

                    writer.writeheader()

                    writer.writerows(list(QUEUE_1S))
                   

                df_30 = process_file(snapshot_path)

                if not df_30.empty:

                    row_30 = df_30.reset_index().iloc[-1].to_dict()
                    QUEUE_30S.append(row_30)
                    QUEUE_1S.clear()

            # -----------------------------------------
            # 3. WAIT FOR 240 30-SECOND OBSERVATIONS
            # -----------------------------------------
            if len(QUEUE_30S) >= 240:
                print("240 historical observations available. Making prediction.")

                # Build historical dataframe directly
                # from the queue.
                history_df = pd.DataFrame(list(QUEUE_30S))

                df = (history_df.set_index("timestamp")
                        .sort_index())

                df = roll_win(df)
                df = preproccesing(df)


                expected_features = set(model.feature_names_in_)
                generated_features = set(df.columns)

                missing = expected_features - generated_features
                extra = generated_features - expected_features

                print("Missing:", missing)
                print("Extra:", extra)

                latest_row = df.iloc[[-1]]
                prediction_results = make_predictions(latest_row)

                # -------------------------------------
                # 6. SAVE PREDICTION TO DATABASE
                # -------------------------------------
                with Session(engine) as session:

                    new_log = Log(
                        timestamp=datetime.fromisoformat(prediction_results["timestamp"]),
                        well_id=well_id,
                        fault_class=prediction_results["prediction"],
                        confidence=prediction_results["confidence"],
                        eng_logged_in=engineer_id
                    )

                    session.add(new_log)
                    session.commit()
                    session.refresh(new_log)


                # -------------------------------------
                # 7. STREAM PREDICTION TO CLIENT
                # -------------------------------------

                yield (json.dumps(prediction_results) + "\n")


                # -------------------------------------
                # 8. SLIDE HISTORY FORWARD
                # -------------------------------------

                QUEUE_30S.popleft()

        print("Simulation completed. No more sensor readings.")

                    
        


def make_predictions(row : pd.DataFrame):
    # -----------------------------------------
    # 1. GET THE TRUE CLASS
    # -----------------------------------------
    # Only available because this is synthetic
    # labelled data used for simulation.

    y_true = int(row["class"].iloc[0])

    timestamp = row.index[0]

    expected_features = list(
        model.feature_names_in_
    )

    # Check that preprocessing actually produced
    # every feature the registered model expects.

    missing_features = [feature for feature in expected_features if feature not in row.columns]


    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
    )


    # Select ONLY model features and preserve
    # the exact training column order.
    X = row[expected_features].copy()

    # -----------------------------------------
    # 4. MAKE PREDICTION
    # -----------------------------------------
    pred_class = model.predict(X)[0]

    probabilities = model.predict_proba(X)

    confidence = probabilities.max(axis=1)[0]

    return {
            "confidence": float(confidence),
            "prediction": int(pred_class),
            "timestamp": timestamp.isoformat(),
            "actual_preds": y_true
        }





@router.get('/live-predict')
def live_predict(current_engineer: Engineer = Depends(get_current_engineer)):


    file_path = (BASE_DIR / "cleaned_csv" / "synthetic_oil_stream_1s_30_predictions.csv")
    well_id = "SYNTH-WELL-001"

    prediction_stream = readings_simulator(file_path, well_id=well_id, engineer_id=current_engineer.engineer_id)

    return StreamingResponse(
        prediction_stream,
        media_type="application/x-ndjson",
        headers={
        "Cache-Control": "no-cache"
        }
    )



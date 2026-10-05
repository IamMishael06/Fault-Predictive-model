from fastapi import FastAPI, Cookie, UploadFile, File, HTTPException, status, Depends, APIRouter
from sqlmodel import Field, Session, SQLModel, create_engine, select
from pydantic import BaseModel
from db.db import engine, Engineer, Log
from auth.security import decode_access_token
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

router = APIRouter(prefix='/predict', tags=['Predictions'])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

class PredictionInput(SQLModel):
    P_PDG : int  = Field(alias="P_PDG", description="Pressure at the pump discharge gauge")
    P_TPT : int = Field(alias="P_TPT", description="Pressure at the pump suction gauge")
    T_TPT : int = Field(alias="T_TPT", description="Temperature at the pump suction gauge")
    P_MON_CKP : int = Field(alias="P_MON_CKP", description="Pressure at the monitor checkpoint")
    T_JUS_CKP : int = Field(alias="T_JUS_CKP", description="Temperature at the justification checkpoint")
    P_JUS_CKGL : int = Field(alias="P_JUS_CKGL", description="Pressure at the justification checkpoint (glow)")
    QGL : int = Field(alias="QGL", description="Flow rate at the glow checkpoint")

def get_current_engineer(token: str = Depends(oauth2_scheme)):
    # Here you would decode the JWT token and retrieve the engineer's information (Here, it is the email)
    payload = decode_access_token(token)
    engineer_id = payload.get("sub")
    print(engineer_id)
    with Session(engine) as session:
        statement = select(Engineer).where(Engineer.engineer_id == engineer_id)
        engineer = session.exec(statement).first()
        if not engineer:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
        return engineer

@router.get('/live-predict')
def live_predict(data : PredictionInput, current_engineer: Engineer = Depends(get_current_engineer)):
    pass
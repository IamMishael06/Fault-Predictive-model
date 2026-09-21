from fastapi import FastAPI, Cookie, UploadFile, File, HTTPException, status, Depends, APIRouter
from sqlmodel import Field, Session, SQLModel, create_engine
from pydantic import BaseModel

router = APIRouter(prefix='/predict', tags=['Predictions'])

@router.post('/live-predict')
def live_predict():
    pass
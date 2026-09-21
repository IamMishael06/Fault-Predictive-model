from fastapi import Body, FastAPI, Cookie, UploadFile, File, HTTPException, status, Depends, APIRouter
from sqlmodel import Field, Session, SQLModel, create_engine, select
from pydantic import BaseModel, StringConstraints
from fastapi.responses import RedirectResponse
from typing import Annotated
from db.db import engine, Engineer
from auth.security import get_password_hash, verify_password, create_access_token
from datetime import datetime, timedelta

router = APIRouter(prefix='/auth', tags=['Authentication'])

class EngineerDeets(SQLModel):
    engineer_id: int = Field(..., description="Unique identifier for the engineer", lt=10**10, gt=10**8)
    engineer_name : str
    engineer_email : Annotated[str, StringConstraints(to_lower=True)]
    engineer_password : str



@router.post('/register', response_model=EngineerDeets)
def register(EngeerDeets : EngineerDeets):
    with Session(engine) as session:
        statement = select(Engineer).where(Engineer.engineer_email == EngeerDeets.engineer_email)
        existing_engineer = session.exec(statement).first()
        if existing_engineer:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Engineer with this email already exists.")
        new_engineer = Engineer(
            engineer_name=EngeerDeets.enginer_name,
            engineer_email=EngeerDeets.engineer_email,
            engineer_password=get_password_hash(EngeerDeets.engineer_password)
        )
        session.add(new_engineer)
        session.commit()
        session.refresh(new_engineer)
        return {"message": f"Engineer registered successfully with ID: {new_engineer.engineer_id} and email: {new_engineer.engineer_email}"}


@router.post('/login')
def login(email : str, password : str):

    custom_time = timedelta(hours=5)
    with Session(engine) as session:
        statement = select(Engineer).where(Engineer.engineer_email == email)
        engineer = session.exec(statement)
        if engineer and verify_password(password, engineer.engineer_password):
            encoded_jwt = create_access_token(data={"email" : engineer.engineer_email}, expires_in=custom_time)
            return {
                "jwt" : encoded_jwt,
                "msg" : f"User Successfully logged {engineer.engineer_email}"
            }
        else:
            return{
                "msg" : "Incorrect Password or email"
            }


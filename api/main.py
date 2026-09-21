from fastapi import FastAPI, Cookie, UploadFile, File, HTTPException, status, Depends
from auth.auth import router as auth_router
from logs.predict import router as logs_router
from sqlmodel import Field, Session, SQLModel, create_engine
from pydantic import BaseModel
from db.db import engine, Engineer
from contextlib import asynccontextmanager


def init_db():
    SQLModel.metadata.create_all(engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()  # Runs right when uvicorn starts
    yield

app = FastAPI(title="Oil Well Fault Detection API", lifespan=lifespan)

# Plug the isolated routers into the main application
app.include_router(auth_router)
app.include_router(logs_router)






@app.get("/")
def root():
    return {"message": "Welcome to the Predictive Maintenance API!"}
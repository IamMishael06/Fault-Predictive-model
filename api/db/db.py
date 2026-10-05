from sqlmodel import SQLModel, create_engine, Session, Field
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, StringConstraints
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

db_password = os.getenv('DB_PASSWORD')

class Engineer(SQLModel, table=True):
    engineer_id: int | None = Field(default=None, primary_key=True, description="Unique identifier for the engineer", lt=10**10, gt=10**8)
    engineer_name: str
    engineer_email: str = Field(index=True)
    engineer_password: str
    created_at : datetime = Field(default_factory=datetime.now, description="Timestamp when the engineer was created")
    is_logged_in : bool = Field(default=False, description="Indicates whether the engineer is currently logged in or not")

class Log(SQLModel, table=True):
    log_id : int  = Field(default=None, primary_key=True, description="Unique identifier for the log", lt=10**10, gt=10**8)
    timestamp : datetime = Field(default_factory=datetime.now, description="Timestamp when the log was created")
    well_id : str = Field(index=True, description="Identifier for the well")
    fault_class : int
    confidence : float 
    eng_logged_in : int = Field(default=None, foreign_key="engineer.engineer_id", description="Foreign key referencing the engineer who logged in")

Database_URL = f"postgresql://postgres:{db_password}@localhost:5432/mars_oil"

engine = create_engine(Database_URL, echo=True)


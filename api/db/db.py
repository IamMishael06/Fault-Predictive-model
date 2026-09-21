from sqlmodel import SQLModel, create_engine, Session, Field
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, StringConstraints
from datetime import datetime

class Engineer(SQLModel, table=True):
    engineer_id: int | None = Field(default=None, primary_key=True, description="Unique identifier for the engineer", lt=10**10, gt=10**8)
    engineer_name: str
    engineer_email: str = Field(index=True)
    engineer_password: str
    created_at : datetime = Field(default_factory=datetime.now, description="Timestamp when the engineer was created")


Database_URL = "postgresql://postgres:mars2006@localhost:5432/mars_oil"

engine = create_engine(Database_URL, echo=True)


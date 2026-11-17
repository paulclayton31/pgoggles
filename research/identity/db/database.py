"""SQLAlchemy engine and session setup for the identity research database."""

import os

import dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

dotenv.load_dotenv()

DATABASE_URL = os.environ["PGOGGLES_DATABASE_URL"]

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


def get_session() -> Session:
    return SessionLocal()

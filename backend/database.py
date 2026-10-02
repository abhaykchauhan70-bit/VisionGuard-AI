"""
database.py
-----------
Sets up the SQLAlchemy engine, session factory and declarative base.
Every route that needs DB access uses the get_db() dependency below.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from backend.config import settings

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True, pool_recycle=3600)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """FastAPI dependency - yields a DB session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Create all tables (run once at startup / via init script)."""
    import backend.models  # noqa: ensures models are registered on Base
    Base.metadata.create_all(bind=engine)

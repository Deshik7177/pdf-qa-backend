from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime

# SQLite URL (easy to use locally; swap this with PostgreSQL URL for prod)
DATABASE_URL = "sqlite:///./pdfs.db"

# Set up the engine and session
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# This is your PDF metadata table
class PDFDocument(Base):
    __tablename__ = "pdf_documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)
    upload_time = Column(DateTime, default=datetime.utcnow)
    text_content = Column(String)  # Store the extracted text

# Function to create tables (run once on startup)
def init_db():
    Base.metadata.create_all(bind=engine)

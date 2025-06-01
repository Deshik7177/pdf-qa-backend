from fastapi import FastAPI
from app.routes import router
from database import init_db  # Import your init_db function

app = FastAPI(
    title="PDF QA Assistant",
    description="Upload PDFs and ask questions based on their content.",
    version="1.0"
)

@app.on_event("startup")
def startup_event():
    init_db()  # This creates your tables if they don't exist

app.include_router(router)

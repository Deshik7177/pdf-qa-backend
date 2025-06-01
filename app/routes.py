from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from app.pdf_utils import extract_text_from_pdf
from app.qa_engine import create_qa_pipeline, get_answer
from database import SessionLocal, PDFDocument
import os
import shutil

router = APIRouter()
UPLOAD_DIR = "app/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

qa_cache = {}
followup_context = {}  # for simple follow-up question context per file (can be enhanced later)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def _get_unique_filename(filename: str) -> str:
    # Prevent overwriting by adding timestamp suffix if file exists
    base, ext = os.path.splitext(filename)
    counter = 1
    unique_name = filename
    while os.path.exists(os.path.join(UPLOAD_DIR, unique_name)):
        unique_name = f"{base}_{counter}{ext}"
        counter += 1
    return unique_name

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed.")

    # Avoid overwriting existing files — save with unique name if needed
    filename = _get_unique_filename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, filename)

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)  # safer streaming copy

        text = extract_text_from_pdf(file_path)

        # Save or update DB record
        pdf_doc = db.query(PDFDocument).filter(PDFDocument.filename == filename).first()
        if pdf_doc:
            pdf_doc.upload_time = datetime.utcnow()
            pdf_doc.text_content = text
        else:
            pdf_doc = PDFDocument(filename=filename, text_content=text)
            db.add(pdf_doc)
        db.commit()

        # Create vector store & chain and cache it for quick QA
        vector_store, chain = create_qa_pipeline(text)
        qa_cache[filename] = (vector_store, chain)
        followup_context[filename] = None  # clear follow-up context on new upload

        return {
            "message": "PDF uploaded and processed successfully!",
            "filename": filename,
            "text_snippet": text[:300]
        }
    except Exception as e:
        # Clean up file on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@router.post("/ask")
async def ask_question(
    filename: str = Query(..., description="Uploaded PDF filename"),
    question: str = Query(..., description="Your question about the PDF content"),
    follow_up: bool = Query(False, description="Is this a follow-up question?"),
    db: Session = Depends(get_db)
):
    # Check if file exists in cache or DB
    if filename not in qa_cache:
        pdf_doc = db.query(PDFDocument).filter(PDFDocument.filename == filename).first()
        if not pdf_doc:
            raise HTTPException(status_code=404, detail="File not found. Upload it first.")
        vector_store, chain = create_qa_pipeline(pdf_doc.text_content)
        qa_cache[filename] = (vector_store, chain)
        followup_context[filename] = None

    vector_store, chain = qa_cache[filename]

    # If follow-up, pass previous context to the QA engine (simulate it)
    prev_context = followup_context.get(filename) if follow_up else None
    answer = get_answer(vector_store, chain, question, context=prev_context)

    # Save current answer as new context for next follow-up
    followup_context[filename] = answer

    return {"answer": answer}

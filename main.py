import os
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

# Import the RAG logic from your utils.py
from utils import process_document, get_rag_chain

# 1. LOAD ENVIRONMENT VARIABLES
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

# 2. SQLITE DATABASE CONFIGURATION
SQLALCHEMY_DATABASE_URL = "sqlite:///./rag_app.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 3. DATABASE MODEL
class ChatSession(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True, index=True) # Using filename as ID for simplicity
    filename = Column(String)
    query_count = Column(Integer, default=0)

# Create tables in rag_app.db
Base.metadata.create_all(bind=engine)

# 4. FASTAPI APP & STATE
app = FastAPI(title="Beginner FastAPI RAG Chatbot")

# Global dictionary to keep FAISS indexes in RAM
# Key: session_id, Value: FAISS vectorstore object
vector_db_registry = {}

# Dependency to get a DB session for each request
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 5. API ROUTES

@app.get("/")
async def root():
    return {"message": "FastAPI RAG Chatbot is running. Visit /docs for API documentation."}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Uploads a PDF, DOCX, or TXT file, creates a FAISS index, 
    and initializes a session in SQLite.
    """
    if not os.path.exists("temp_storage"):
        os.makedirs("temp_storage")
    
    file_path = f"temp_storage/{file.filename}"
    
    # Save the file locally
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Generate the FAISS vectorstore from the document
        vectorstore = process_document(file_path)
        vector_db_registry[file.filename] = vectorstore
        
        # Save or update session in SQLite
        new_session = ChatSession(id=file.filename, filename=file.filename, query_count=0)
        db.merge(new_session) 
        db.commit()
        
        return {
            "message": "File processed and indexed successfully", 
            "session_id": file.filename
        }
    except Exception as e:
        # Cleanup file if processing fails
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/sessions/")
async def list_sessions(db: Session = Depends(get_db)):
    """Fetches all uploaded document sessions and their query counts."""
    sessions = db.query(ChatSession).all()
    return [
        {
            "id": s.id, 
            "filename": s.filename, 
            "queries_used": s.query_count, 
            "queries_left": 10 - s.query_count
        } for s in sessions
    ]

@app.post("/chat/")
async def chat(session_id: str, query: str, db: Session = Depends(get_db)):
    """
    Performs RAG based on the document. Limited to 10 queries per session.
    """
    # Verify session exists in DB and RAM
    session_data = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    
    if not session_data or session_id not in vector_db_registry:
        raise HTTPException(status_code=404, detail="Session not found. Please upload the file again.")
    
    # Check query limit
    if session_data.query_count >= 10:
        raise HTTPException(status_code=403, detail="Query limit reached (Max 10). Please delete and re-upload.")

    # Retrieve vectorstore and get the RAG chain
    vectorstore = vector_db_registry[session_id]
    rag_chain = get_rag_chain(vectorstore, GROQ_API_KEY)
    
    # Execute RAG query
    response = rag_chain.invoke({"input": query})
    
    # Increment query count in database
    session_data.query_count += 1
    db.commit()
    
    # Extract unique sources from metadata
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in response["context"]]))
    
    return {
        "answer": response["answer"],
        "sources": sources,
        "queries_remaining": 10 - session_data.query_count
    }

@app.delete("/session/{session_id}")
async def delete_session(session_id: str, db: Session = Depends(get_db)):
    """Deletes the document file, the database record, and the memory index."""
    session_data = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found.")

    # 1. Remove physical file
    file_path = f"temp_storage/{session_data.filename}"
    if os.path.exists(file_path):
        os.remove(file_path)

    # 2. Remove from RAM registry
    if session_id in vector_db_registry:
        del vector_db_registry[session_id]

    # 3. Delete from SQLite
    db.delete(session_data)
    db.commit()

    return {"message": f"Session '{session_id}' deleted successfully."}

if __name__ == "__main__":
    import uvicorn
    # Start the server
    uvicorn.run(app, host="127.0.0.1", port=8000)
    
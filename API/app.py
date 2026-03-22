from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import sys
import os
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RAG.rag import (
    load_documents, split_documents,
    create_vector_store, load_vector_store,
    build_rag_chain, query
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain, vector_store
    rag_chain = None
    vector_store = None
    
    persist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_db"))
    
    try:
        if os.path.exists(persist_dir):
            logger.info(f"Loading vector store from {persist_dir}")
            vector_store = load_vector_store(persist_dir)
            count = vector_store._collection.count()
            logger.info(f"Loaded vector store with {count} chunks")
            if count == 0:
                logger.warning("Vector store is empty. Removing and recreating...")
                import shutil
                shutil.rmtree(persist_dir, ignore_errors=True)
                docs = load_documents()
                chunks = split_documents(docs)
                vector_store = create_vector_store(chunks, persist_dir)
        else:
            logger.info(f"Creating new vector store at {persist_dir}")
            docs = load_documents()
            chunks = split_documents(docs)
            vector_store = create_vector_store(chunks, persist_dir)
        
        logger.info(f"Vector store ready with {vector_store._collection.count()} chunks")
    except Exception as e:
        logger.error(f"Failed to setup vector store: {str(e)}")
        raise
    
    rag_chain = build_rag_chain(vector_store, GROQ_API_KEY)
    logger.info("RAG pipeline ready")
    
    yield
    
    # Cleanup on shutdown if needed
    logger.info("RAG pipeline shutdown")

app = FastAPI(title="Energy Audit RAG Assistant", lifespan=lifespan)


class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]

@app.get("/")
def root():
    return {"message": "Energy Audit RAG Assistant is running"}

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    global vector_store  # Make available from startup
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    result = query(rag_chain, vector_store, request.question)
    return AnswerResponse(
        answer=result["answer"],
        sources=result["sources"]
    )

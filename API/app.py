from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RAG.rag import (
    load_documents, split_documents,
    create_vector_store, load_vector_store,
    build_rag_chain, query
)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set")

app = FastAPI(title="Energy Audit RAG Assistant")

@app.on_event("startup")
async def startup():
    global rag_chain
    rag_chain = None
    # Startup
    if not os.path.exists("chroma_db/"):
        docs = load_documents()
        chunks = split_documents(docs)
        vector_store = create_vector_store(chunks)
    else:
        vector_store = load_vector_store()
    
    rag_chain = build_rag_chain(vector_store, GROQ_API_KEY)
    print("RAG pipeline ready")

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
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    result = query(rag_chain, request.question)
    return AnswerResponse(
        answer=result["answer"],
        sources=result["sources"]
    )
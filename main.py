from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI
from pydantic import BaseModel

# Import your custom modules
from backend.qdrant_client import retrieve_context
from backend.neon_client import generate_answer

app = FastAPI(title="Physical AI RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any domain; you can replace "*" with your frontend URL later
    allow_credentials=True,
    allow_methods=["*"],  # Allow GET, POST, OPTIONS, etc.
    allow_headers=["*"],  # Allow any headers
)

class Message(BaseModel):
    message: str

@app.post("/api/chat")
async def chat(msg: Message):
    # Step 1: Retrieve relevant context from book
    context = retrieve_context(msg.message)
    
    # Step 2: Generate answer using OpenAI Agents
    answer = generate_answer(msg.message, context)
    
    return {"answer": answer}

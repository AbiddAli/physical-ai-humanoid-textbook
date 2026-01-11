import os
import logging
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import cohere

# --------------------------------------------------
# ENV & LOGGING
# --------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "textbook_chapters")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not all([QDRANT_URL, QDRANT_API_KEY, COHERE_API_KEY]):
    raise RuntimeError("❌ Missing environment variables")

# --------------------------------------------------
# INIT CLIENTS
# --------------------------------------------------
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
co = cohere.Client(COHERE_API_KEY)

# --------------------------------------------------
# FASTAPI
# --------------------------------------------------
app = FastAPI(title="AI Tutor Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# SCHEMAS
# --------------------------------------------------
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]

# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.get("/")
def root():
    return {"status": "running"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        # 1️⃣ Embed question
        try:
            query_vector = embedder.encode(req.question).tolist()
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return ChatResponse(
                answer="I'm having trouble processing your question. Please try again.",
                sources=[]
            )

        # 2️⃣ Query Qdrant WITHOUT FILTERS (SAFE)
        points = []
        try:
            points = qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=10,
                with_payload=True
            )
        except Exception as e:
            logger.error(f"Qdrant query failed: {e}")
            logger.error(traceback.format_exc())
            return ChatResponse(
                answer="I'm having trouble searching the knowledge base. Please try again later.",
                sources=[]
            )

        if not points:
            return ChatResponse(
                answer="No relevant content found in the textbook.",
                sources=[]
            )

        # 3️⃣ Build context from payload
        contexts = []
        sources = set()
        for point in points:
            try:
                payload = getattr(point, 'payload', {}) or {}
                
                text = payload.get("page_content") or payload.get("text")
                source = payload.get("source", "unknown")
                           

                if text:
                    contexts.append(text)
                    sources.add(source)
            except Exception as e:
                logger.warning(f"Error processing point payload: {e}")
                continue

        if not contexts:
            return ChatResponse(
                answer="No relevant content found in the textbook.",
                sources=[]
            )

        context = "\n\n".join(contexts)

        # 4️⃣ Cohere Chat
        try:
            response = co.chat(
                model="command-r-08-2024",
                message=f"""
You are an AI tutor.
Answer ONLY using the context below.

Context:
{context}

Question:
{req.question}

If the answer is not in the context, say so clearly.
""",
                temperature=0.3,
            )
            answer = getattr(response, 'text', None) or getattr(response, 'message', None) or \
                     "I'm having trouble generating a response. Please try again."
        except Exception as e:
            logger.error(f"Cohere API failed: {e}")
            logger.error(traceback.format_exc())
            return ChatResponse(
                answer="I'm having trouble generating a response. The AI service may be temporarily unavailable.",
                sources=list(sources) if sources else []
            )

        return ChatResponse(answer=answer, sources=list(sources))

    except Exception as e:
        logger.exception("Unexpected error in chat endpoint")
        return ChatResponse(
            answer="An unexpected error occurred. Please try again later.",
            sources=[]
        )

# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------
@app.get("/health")
def health():
    health_status = {"qdrant": False, "embedder": False, "cohere": False, "collection": COLLECTION_NAME}
    try:
        qdrant.get_collection(COLLECTION_NAME)
        health_status["qdrant"] = True
    except Exception as e:
        logger.warning(f"Qdrant health check failed: {e}")

    try:
        embedder.encode("test").tolist()
        health_status["embedder"] = True
    except Exception as e:
        logger.warning(f"Embedder health check failed: {e}")

    try:
        health_status["cohere"] = bool(co)
    except Exception as e:
        logger.warning(f"Cohere health check failed: {e}")

    return health_status
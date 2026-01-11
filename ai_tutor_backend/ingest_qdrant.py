import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import uuid

# ------------------------------------------------------------------
# LOAD ENV
# ------------------------------------------------------------------
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "textbook_chapters")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("❌ QDRANT_URL or QDRANT_API_KEY missing")

# ------------------------------------------------------------------
# INIT CLIENT & EMBEDDER
# ------------------------------------------------------------------
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ------------------------------------------------------------------
# SAMPLE DATA
# ------------------------------------------------------------------
documents = [
    {
        "text": "Artificial Intelligence is the simulation of human intelligence by machines.",
        "source": "chapter1"
    },
    {
        "text": "Robotics combines AI with mechanical engineering to build autonomous systems.",
        "source": "chapter2"
    }
]

# ------------------------------------------------------------------
# CREATE COLLECTION IF MISSING
# ------------------------------------------------------------------
existing = [c.name for c in qdrant.get_collections().collections]
if COLLECTION_NAME not in existing:
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"size": 384, "distance": "Cosine"}  # MiniLM-L6-v2 embeddings
    )
    print(f"✅ Collection '{COLLECTION_NAME}' created")

# ------------------------------------------------------------------
# EMBED & UPLOAD
# ------------------------------------------------------------------
points = []
for doc in documents:
    vector = embedder.encode(doc["text"]).tolist()
    # Use UUID for point ID
    point_id = str(uuid.uuid4())
    points.append({
        "id": point_id,
        "vector": vector,
        "payload": {
            "page_content": doc["text"],
            "source": doc["source"]
        }
    })

qdrant.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print(f"✅ {len(points)} documents ingested into Qdrant collection '{COLLECTION_NAME}'")

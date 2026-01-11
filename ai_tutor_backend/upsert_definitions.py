"""
Script to upsert Physical AI definition chunks to existing Qdrant collection
"""
import json
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid

# Load environment
load_dotenv()

# Initialize Qdrant client
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "textbook_chapters")

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Load the embedded definitions
with open('embedded_physical_ai_definitions.json', 'r') as f:
    embedded_definitions = json.load(f)

print(f"Loaded {len(embedded_definitions)} definition chunks to upsert")

# Get current collection info before upsert
try:
    collection_info = qdrant.get_collection(COLLECTION_NAME)
    initial_count = collection_info.points_count
    print(f"Current collection size: {initial_count}")
except Exception as e:
    print(f"Could not get collection info: {e}")
    initial_count = 0

# Upsert the new definition chunks to the existing collection
try:
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=embedded_definitions
    )
    print(f"Successfully upserted {len(embedded_definitions)} new definition chunks to collection '{COLLECTION_NAME}'")
except Exception as e:
    print(f"Error during upsert: {e}")
    raise

# Verify the collection size after upsert
try:
    collection_info = qdrant.get_collection(COLLECTION_NAME)
    final_count = collection_info.points_count
    print(f"New collection size: {final_count}")
    print(f"Collection size increased by: {final_count - initial_count}")
except Exception as e:
    print(f"Could not get collection info after upsert: {e}")

print("Definition chunks successfully added to Qdrant collection!")
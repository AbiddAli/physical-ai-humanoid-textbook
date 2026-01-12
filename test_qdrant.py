from backend.qdrant_client import retrieve_context

query = "What is Physical AI?"
print(retrieve_context(query))

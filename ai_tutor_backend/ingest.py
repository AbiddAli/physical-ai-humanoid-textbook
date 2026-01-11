import os
import glob
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Paths & config
qdrant_path = os.getenv("QDRANT_PATH", "./qdrant_data")
textbook_path = os.getenv("TEXTBOOK_PATH", "../docs")
collection_name = os.getenv("COLLECTION_NAME", "textbook_chapters")


def load_markdown_files():
    """Load all markdown files from the textbook directory."""
    print(f"Loading markdown files from: {textbook_path}")

    md_files = glob.glob(f"{textbook_path}/**/*.md", recursive=True)

    documents = []
    for file_path in md_files:
        print(f"Loading: {file_path}")
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = file_path
                doc.metadata["filename"] = os.path.basename(file_path)

            documents.extend(docs)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return documents


def chunk_documents(documents):
    """Chunk documents into smaller overlapping pieces."""
    print("Chunking documents...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunked_documents = []
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            chunked_documents.append(
                Document(
                    page_content=chunk,
                    metadata=doc.metadata
                )
            )

    print(f"Created {len(chunked_documents)} chunks")
    return chunked_documents


def create_vector_store(documents):
    """Create and persist Qdrant vector store."""
    print("Creating vector store...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    Qdrant.from_documents(
        documents=documents,
        embedding=embeddings,
        path=qdrant_path,
        collection_name=collection_name,
    )

    print(f"Vector store saved at: {qdrant_path}")


def main():
    print("Starting ingestion pipeline...")

    documents = load_markdown_files()
    if not documents:
        print("No markdown files found.")
        return

    chunked_documents = chunk_documents(documents)
    create_vector_store(chunked_documents)

    print("Ingestion completed successfully!")


if __name__ == "__main__":
    main()

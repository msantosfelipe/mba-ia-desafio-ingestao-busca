import os
import warnings
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_postgres import PGVector


# Ignore deprecation warnings from OllamaEmbeddings
warnings.filterwarnings(
    "ignore",
    message=".*OllamaEmbeddings.*deprecated.*"
)

load_dotenv()


SPLIT_PDF_CHUNK_SIZE = 1000
SPLIT_PDF_CHUNK_OVERLAP = 200

PDF_PATH = os.getenv("PDF_PATH")
USE_OLLAMA = os.getenv("USE_OLLAMA", "False").lower() == "true"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
DATABASE_URL = os.getenv("DATABASE_URL")
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")



def ingest_pdf():
    documents = _load_pdf()
    if not documents:
        raise ValueError("No documents were loaded from the PDF.")

    chunks = _split_documents_in_chunks(documents)
    print(f"Number of chunks created: {len(chunks)}")

    _embed_and_save_chunks(chunks)

def _load_pdf():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    return documents

def _split_documents_in_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=SPLIT_PDF_CHUNK_SIZE,
        chunk_overlap=SPLIT_PDF_CHUNK_OVERLAP,
        separators=["\n\n", "\n", "", " "]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def _embed_and_save_chunks(chunks):
    enriched = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
        )
        for d in chunks
    ]
    ids = [f"doc-{i}" for i in range(len(enriched))]

    if USE_OLLAMA:
        embeddings = OllamaEmbeddings(
            model=OLLAMA_MODEL_NAME,
            base_url=OLLAMA_BASE_URL
        )
    else:
        embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
    
    store = PGVector(
        embeddings=embeddings,
        collection_name=PG_VECTOR_COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )
    
    # add or update
    store.add_documents(enriched, ids=ids)
    print(f"Added {len(enriched)} documents to PGVector store.")


if __name__ == "__main__":
    print("Starting PDF ingestion process...")
    ingest_pdf()

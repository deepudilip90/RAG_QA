import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight, good for local use
PATH_DOC_CHUNKS = 'extracted_pdfs/extracts.json'
PATH_PERSISTENT_VECTOR_STORE = "vector_store/faiss_index.bin"


def _generate_embeddings(doc_chunks):
    embeddings = EMBEDDING_MODEL.encode([chunk["text"] for chunk in doc_chunks], show_progress_bar=True)
    print(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    return embeddings


def _prepare_faiss_store(embeddings):
    dimension = embeddings.shape[1]  # Embedding size (e.g., 384)
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
    index.add(embeddings.astype(np.float32))  # Add embeddings to index
    faiss.write_index(index, PATH_PERSISTENT_VECTOR_STORE )  # Save index to disk


def create_and_store_embeddings():
    with open(PATH_DOC_CHUNKS, 'r') as fp:
        doc_chunks = json.load(fp)

    embeddings = _generate_embeddings(doc_chunks)
    _prepare_faiss_store(embeddings)


if __name__ == "__main__":
    create_and_store_embeddings()
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import json


def extract_text_from_pdfs(pdf_folder):
    documents = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf") and filename != "The Ordinary Heroes of the Taj.pdf":
            filepath = os.path.join(pdf_folder, filename)
            with pdfplumber.open(filepath) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                documents.append({"text": text, "source": filename})
    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Max characters per chunk
        chunk_overlap=200  # Overlap for context
    )
    chunks = []
    for doc in documents:
        split_texts = text_splitter.split_text(doc["text"])
        for i, text in enumerate(split_texts):
            chunks.append({"text": text, "source": doc["source"], "chunk_id": i})
    return chunks


def extract_pdfs():
    #    Example usage
    pdf_folder = "pdfs/"
    raw_docs = extract_text_from_pdfs(pdf_folder)
    doc_chunks = split_documents(raw_docs)
    print(f"Extracted {len(doc_chunks)} chunks from {len(raw_docs)} PDFs")
    with open("extracted_pdfs/extracts.json", 'w') as fp:
        json.dump(doc_chunks, fp)


if __name__ == '__main__':
    extract_pdfs()
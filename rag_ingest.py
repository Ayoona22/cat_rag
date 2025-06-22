# rag_ingest.py (multi-PDF)

import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# 1. Extract text
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

# 2. Chunk text
def split_into_chunks(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# 3. Store in Chroma
def embed_and_store(chunks, source_id, collection):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks).tolist()

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i]],
            ids=[f"{source_id}_{i}"]
        )
    print(f"✅ Stored {len(chunks)} chunks from {source_id}")

from chromadb.config import Settings
from chromadb import PersistentClient

# 4. Process all PDFs in folder
def process_all_pdfs(folder_path):
    
    chroma_client = PersistentClient(path="./rag_store")
    collection = chroma_client.get_or_create_collection("rag_documents")

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            base_name = os.path.splitext(filename)[0]  # e.g. "cat_notes"
            text = extract_text_from_pdf(file_path)
            chunks = split_into_chunks(text)
            embed_and_store(chunks, base_name, collection)

    #chroma_client.persist()
    print("🎉 All PDFs ingested successfully!")

# Run
if __name__ == "__main__":
    pdf_folder = "resources/"  # Put all your PDFs here
    process_all_pdfs(pdf_folder)

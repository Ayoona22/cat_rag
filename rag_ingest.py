import os
import re
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text.strip())
    
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
    text = re.sub(r'(\d+)([A-Z])', r'\1 \2', text)    # Add space between numbers and letters
    
    text = re.sub(r'\n\d+\n', '\n', text)  # Remove standalone page numbers
    text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
    
    return text

def extract_text_from_pdf(pdf_path):
    print(f"Processing PDF: {pdf_path}")
    try:
        reader = PdfReader(pdf_path)
        text_parts = []
        
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():  # Only add non-empty pages
                text_parts.append(f"[Page {i+1}] {page_text}")
        
        full_text = "\n".join(text_parts)
        return preprocess_text(full_text)
    
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return ""

def smart_chunk_text(text, chunk_size=800, chunk_overlap=100):
    # First, try to split by sections/chapters if they exist
    section_patterns = [
        r'\n(?:Chapter|CHAPTER)\s+\d+',
        r'\n(?:Section|SECTION)\s+\d+',
        r'\n\d+\.\s+[A-Z][^.]*\n',  # Numbered sections
        r'\n[A-Z][A-Z\s]{10,}\n',   # ALL CAPS headers
    ]
    pattern = "|".join(section_patterns)
    sections = re.split(pattern, text)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n","\n",". ",", "," ",""],            
        length_function=len,
    )
    filtered_chunks = []
    for section in sections:
        chunks = text_splitter.split_text(section)
        for chunk in chunks:
            chunk = chunk.strip()
            if 50 <= len(chunk) <= 2000:
                filtered_chunks.append(chunk)

    return filtered_chunks

def embed_and_store(chunks, source_id, collection):
    if not chunks:
        print(f"No valid chunks found for {source_id}")
        return
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    batch_size = 50
    total_stored = 0
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        
        try:
            embeddings = model.encode(batch_chunks).tolist()
            
            metadatas = []
            ids = []
            
            for j, chunk in enumerate(batch_chunks):
                chunk_id = f"{source_id}_chunk_{i+j}"
                metadata = {
                    "source": source_id,
                    "chunk_index": i + j,
                    "chunk_length": len(chunk),
                    "chunk_preview": chunk[:100] + "..." if len(chunk) > 100 else chunk
                }
                
                metadatas.append(metadata)
                ids.append(chunk_id)
            
            # Store in ChromaDB
            collection.add(
                documents=batch_chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            total_stored += len(batch_chunks)
            print(f"Stored batch {i//batch_size + 1}: {len(batch_chunks)} chunks")
            
        except Exception as e:
            print(f"Error storing batch for {source_id}: {str(e)}")
            continue
    
    print(f"✅ Total stored for {source_id}: {total_stored} chunks")

def process_all_pdfs(folder_path):
    """Process all PDFs in the specified folder"""
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist!")
        return
    
    chroma_client = PersistentClient(path="./rag_store")
    try:
        chroma_client.delete_collection("rag_documents")
        print("Deleted existing collection")
    except:
        pass
    
    collection = chroma_client.get_or_create_collection(
        name="rag_documents",
        metadata={"description": "JEE Chemistry study materials"}
    )
    
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in '{folder_path}'")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    total_chunks = 0
    processed_files = 0
    
    for filename in pdf_files:
        file_path = os.path.join(folder_path, filename)
        base_name = os.path.splitext(filename)[0]
        print(f"\n--- Processing: {filename} ---")
        
        text = extract_text_from_pdf(file_path)
        
        if not text or len(text) < 100:
            print(f"⚠️  Skipping {filename} - insufficient text content")
            continue
        
        chunks = smart_chunk_text(text)
        if not chunks:
            print(f"⚠️  No valid chunks created for {filename}")
            continue
        
        embed_and_store(chunks, base_name, collection)
        
        total_chunks += len(chunks)
        processed_files += 1
    
    print(f"\n🎉 Processing complete!")
    print(f"📊 Statistics:")
    print(f"   - Files processed: {processed_files}/{len(pdf_files)}")
    print(f"   - Total chunks created: {total_chunks}")
    print(f"   - Average chunks per file: {total_chunks/processed_files if processed_files > 0 else 0:.1f}")

def verify_ingestion():
    try:
        chroma_client = PersistentClient(path="./rag_store")
        collection = chroma_client.get_collection("rag_documents")
        
        count = collection.count()
        print(f"\n✅ Verification: {count} documents stored in the vector database")
        
        if count > 0:
            sample = collection.peek(limit=1)
            if sample['documents']:
                print(f"📝 Sample chunk preview: {sample['documents'][0][:200]}...")
        
        return count > 0
    
    except Exception as e:
        print(f"❌ Verification failed: {str(e)}")
        return False

if __name__ == "__main__":
    pdf_folder = "resources/"  
    
    print("🚀 Starting RAG ingestion for JEE Chemistry materials...")
    print(f"📁 Looking for PDFs in: {pdf_folder}")
    
    process_all_pdfs(pdf_folder)
    verify_ingestion()
    
    

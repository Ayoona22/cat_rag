import os
import re
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
import base64

from typing_extensions import TypedDict
from typing import Optional

class IngestState(TypedDict):
    file_path: str
    file_name: str
    raw_text: Optional[str]
    cleaned_text: Optional[str]
    chunks: Optional[list[str]]
    error: Optional[str]
    image_data: Optional[list[dict]]

def load_pdf_node(state: IngestState) -> IngestState:
    try:
        doc = fitz.open(state["file_path"])
        full_text = []
        image_refs = []

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            full_text.append(f"[Page {page_num}] {text.strip()}")

            # Extract images from page
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # OCR the image to capture any embedded text (like equations)
                try:
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    ocr_text = pytesseract.image_to_string(image).strip()
                except Exception as ocr_err:
                    ocr_text = ""

                # Save image as base64 (optional – you can store or skip)
                encoded_img = base64.b64encode(image_bytes).decode("utf-8")

                # Append OCR text + a tag to insert image metadata
                if ocr_text:
                    full_text.append(f"\n[Image_OCR_Page_{page_num}_{img_index}]: {ocr_text}\n")

                image_refs.append({
                    "page": page_num,
                    "index": img_index,
                    "ocr_text": ocr_text,
                    "image_format": image_ext,
                    "base64_data": encoded_img
                })

        # Final state update
        state["raw_text"] = "\n".join(full_text)
        state["image_data"] = image_refs  # Optional: add this to IngestState if needed

    except Exception as e:
        state["error"] = f"PDF read error: {str(e)}"
    return state


def preprocess_node(state: IngestState) -> IngestState:
    if not state.get("raw_text"):
        state["error"] = "No raw text to preprocess"
        return state

    text = state["raw_text"]
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'(\d+)([A-Z])', r'\1 \2', text)
    text = re.sub(r'\n\d+\n', '\n', text)
    text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
    state["cleaned_text"] = text
    return state


def chunk_node(state: IngestState) -> IngestState:
    if not state.get("cleaned_text"):
        state["error"] = "No cleaned text for chunking"
        return state

    section_patterns = [
        r'\n(?:Chapter|CHAPTER)\s+\d+',
        r'\n(?:Section|SECTION)\s+\d+',
        r'\n\d+\.\s+[A-Z][^.]*\n',
        r'\n[A-Z][A-Z\s]{10,}\n',
    ]
    pattern = "|".join(section_patterns)
    sections = re.split(pattern, state["cleaned_text"])
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        length_function=len,
    )

    all_chunks = []
    for section in sections:
        chunks = splitter.split_text(section.strip())
        all_chunks.extend(chunk.strip() for chunk in chunks if 50 <= len(chunk) <= 2000)

    state["chunks"] = all_chunks
    return state


def embed_store_node(state: IngestState) -> IngestState:
    if not state.get("chunks"):
        state["error"] = "No chunks to embed"
        return state

    model = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = PersistentClient(path="./rag_store")
    collection = chroma_client.get_or_create_collection("rag_documents")

    batch_size = 50
    for i in range(0, len(state["chunks"]), batch_size):
        batch = state["chunks"][i:i + batch_size]
        try:
            embeddings = model.encode(batch).tolist()
            ids = [f"{state['file_name']}_chunk_{i+j}" for j in range(len(batch))]
            metadatas = [{
                "source": state["file_name"],
                "chunk_index": i + j,
                "chunk_length": len(chunk),
                "chunk_preview": chunk[:100] + "..." if len(chunk) > 100 else chunk
            } for j, chunk in enumerate(batch)]

            collection.add(
                documents=batch,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            state["error"] = f"Embedding error: {str(e)}"
            return state

    return state


from langgraph.graph import StateGraph, END

def create_ingest_graph():
    builder = StateGraph(IngestState)
    builder.add_node("load_pdf", load_pdf_node)
    builder.add_node("preprocess", preprocess_node)
    builder.add_node("chunk", chunk_node)
    builder.add_node("embed_store", embed_store_node)

    builder.set_entry_point("load_pdf")
    builder.add_edge("load_pdf", "preprocess")
    builder.add_edge("preprocess", "chunk")
    builder.add_edge("chunk", "embed_store")
    builder.add_edge("embed_store", END)

    return builder.compile()


if __name__ == "__main__":
    folder_path = "resources/"
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    graph = create_ingest_graph()

    for file in pdf_files:
        file_path = os.path.join(folder_path, file)
        print(f"📄 Processing {file}...")

        initial_state = {
            "file_path": file_path,
            "file_name": os.path.splitext(file)[0],
            "raw_text": None,
            "cleaned_text": None,
            "chunks": None,
            "error": None,
        }

        final_state = graph.invoke(initial_state)
        if not final_state["error"]:
            output_path = f"outputs/{file}.txt"
            os.makedirs("outputs", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_state["raw_text"] or "")
            print(f"✅ Text saved to: {output_path}")

    

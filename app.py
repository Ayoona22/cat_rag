from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import Annotated, Dict, Any, Optional
from typing_extensions import TypedDict
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import pytesseract
import speech_recognition as sr
from PIL import Image
import os
from dotenv import load_dotenv
from models import (
    save_user_question,
    find_similar_question,
    save_session,
    session_exists,
    get_last_n_messages
)
from datetime import datetime
from PyPDF2 import PdfReader
import re

SIMILARITY_THRESHOLD = 0.85  

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
embedder = SentenceTransformer('all-MiniLM-L6-v2')

if os.name == 'nt':  
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class ChatState(TypedDict):
    session_id: str
    user_input: str
    input_type: str  
    file_data: Optional[Any]
    extracted_text: str
    embedding: Optional[list]
    cached_answer: Optional[str]
    final_answer: str
    error_message: Optional[str]
    user_preference: Optional[str]

def preprocess_text(text):
    """Clean and preprocess text for better embeddings"""
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s\.\,\?\!\-\(\)]', ' ', text)
    return text

def check_input_type(state: ChatState) -> ChatState:
    print("Checking input type...")
    if state.get("file_data"):
        file_type = state["file_data"].get("type", "")
        if file_type.startswith('image/'):
            state["input_type"] = "image"
        elif file_type.startswith('audio/'):
            state["input_type"] = "audio"
        elif file_type == 'application/pdf':
            state["input_type"] = "pdf"
        else:
            state["input_type"] = "text"
    else:
        state["input_type"] = "text"
    return state

def extract_text_from_pdf(state: ChatState) -> ChatState:
    print("Extracting text from PDF...")
    try:
        file_data = state["file_data"]
        reader = PdfReader(file_data["stream"])
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        processed_text = preprocess_text(text)
        state["extracted_text"] = f"[PDF Content]: {processed_text or 'No readable text found'}"
    except Exception as e:
        state["error_message"] = f"PDF processing error: {str(e)}"
        state["extracted_text"] = "[PDF Content]: Could not extract text."
    return state

def extract_text_from_image(state: ChatState) -> ChatState:
    print("Extracting text from image...")
    try:
        file_data = state["file_data"]
        image = Image.open(file_data["stream"])
        
        image = image.convert('L')  
        
        custom_config = r'--oem 3 --psm 6'
        extracted = pytesseract.image_to_string(image, config=custom_config)
        
        processed_text = preprocess_text(extracted)
        state["extracted_text"] = f"[Image Content]: {processed_text or 'No readable text found'}"
        
        if not processed_text:
            custom_config = r'--oem 3 --psm 11'
            extracted = pytesseract.image_to_string(image, config=custom_config)
            processed_text = preprocess_text(extracted)
            state["extracted_text"] = f"[Image Content]: {processed_text or 'No readable text found'}"
            
    except Exception as e:
        state["error_message"] = f"Image processing error: {str(e)}"
        state["extracted_text"] = "[Image Content]: Could not extract text."
    return state

def extract_text_from_audio(state: ChatState) -> ChatState:
    print("Extracting text from audio...")
    try:
        recognizer = sr.Recognizer()
        file_data = state["file_data"]
        
        # Adjust for ambient noise
        with sr.AudioFile(file_data["stream"]) as source:
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.record(source)
            
        text = recognizer.recognize_google(audio_data)
        processed_text = preprocess_text(text)
        state["extracted_text"] = f"[Audio Content]: {processed_text}"
        
    except sr.UnknownValueError:
        state["extracted_text"] = "[Audio Content]: Could not understand the audio."
    except sr.RequestError as e:
        state["error_message"] = f"Audio recognition service error: {str(e)}"
        state["extracted_text"] = "[Audio Content]: Recognition service unavailable."
    except Exception as e:
        state["error_message"] = f"Audio processing error: {str(e)}"
        state["extracted_text"] = "[Audio Content]: Could not extract text."
    return state

def embed_text(state: ChatState) -> ChatState:
    print("Creating embedding for the text...")
    full_text = state["user_input"] or ""
    if state.get("extracted_text"):
        full_text = f"{state['extracted_text']}\n{full_text}" if state["user_input"] else state["extracted_text"]
    
    full_text = preprocess_text(full_text)
    
    try:
        if full_text:
            embedding = embedder.encode(full_text)
            state["embedding"] = embedding.tolist()
    except Exception as e:
        state["error_message"] = f"Embedding error: {str(e)}"
    return state

def check_cache(state: ChatState) -> ChatState:
    print("Checking cache for similar questions...")
    try:
        if state.get("embedding"):
            cached_answer = find_similar_question(
                state["session_id"], 
                state["embedding"], 
                SIMILARITY_THRESHOLD
            )
            state["cached_answer"] = cached_answer
    except Exception as e:
        state["error_message"] = f"Cache check error: {str(e)}"
        state["cached_answer"] = None
    return state

def return_cached_answer(state: ChatState) -> ChatState:
    print("Returning cached answer...")
    state["final_answer"] = state["cached_answer"]
    return state

from chromadb import PersistentClient

chroma_client = PersistentClient(path="./rag_store")
collection = chroma_client.get_or_create_collection("rag_documents")

def generate_answer(state: ChatState) -> ChatState:
    print("Generating new answer using Gemini + RAG...")

    try:
        if not session_exists(state["session_id"]):
            save_session(state["session_id"], datetime.utcnow())

        full_input = state["user_input"] or ""
        if state.get("extracted_text"):
            full_input = f"{state['extracted_text']}\n{full_input}" if full_input else state["extracted_text"]

        full_input = preprocess_text(full_input)

        query_embedding = embedder.encode(full_input).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,  
            include=['documents', 'distances']  
        )
        
        context_chunks = results['documents'][0] if results['documents'] else []
        distances = results['distances'][0] if results['distances'] else []
        
        relevant_chunks = []
        for chunk, distance in zip(context_chunks, distances):
            if distance < 0.7: 
                relevant_chunks.append(chunk)
        
        if not relevant_chunks:
            relevant_chunks = context_chunks[:2]
        
        context_text = "\n\n".join(relevant_chunks)
        print(context_text)  
        

        instruction_text = (
            "- Just give the final answer, no explanation."
            if state.get("user_preference") == "short"
            else """- Provide a clear, detailed explanation.
        - Include relevant formulas, equations, or concepts.
        - If it's a problem, show step-by-step solution.
        - Reference specific concepts from the study material when applicable.
        - If the question is not chemistry-related or cannot be answered from the material, politely explain the limitation."""
        )

        prompt = f"""You are an expert chemistry tutor specializing in JEE preparation. 
        Use the following study material to answer the student's question.

        Study Material:
        {context_text}

        Student's Question:
        {full_input}

        Instructions:
        {instruction_text}

        Answer:"""


        response = model.generate_content(prompt)
        answer = response.text

        save_user_question(
            state["session_id"],
            full_input,
            answer,
            query_embedding
        )

        state["final_answer"] = answer

    except Exception as e:
        print(f"RAG error details: {str(e)}")
        state["error_message"] = f"RAG error: {str(e)}"
        state["final_answer"] = "Sorry, an error occurred while generating your answer. Please try again."

    return state

def route_after_input_check(state: ChatState) -> str:
    input_type = state.get("input_type", "text")
    if input_type == "image":
        return "extract_image"
    elif input_type == "audio":
        return "extract_audio"
    elif input_type == "pdf":
        return "extract_pdf"
    else:
        return "embed_text"

def route_after_cache_check(state: ChatState) -> str:
    if state.get("cached_answer"):
        return "return_cached"
    else:
        return "generate_new"

def create_chat_workflow():
    workflow = StateGraph(ChatState)
    workflow.add_node("check_input", check_input_type)
    workflow.add_node("extract_image", extract_text_from_image)
    workflow.add_node("extract_audio", extract_text_from_audio)
    workflow.add_node("extract_pdf", extract_text_from_pdf)
    workflow.add_node("embed_text", embed_text)
    # workflow.add_node("check_cache", check_cache)
    # workflow.add_node("return_cached", return_cached_answer)
    workflow.add_node("generate_new", generate_answer)
    workflow.set_entry_point("check_input")    
    workflow.add_conditional_edges(
        "check_input",
        route_after_input_check,
        {
            "extract_image": "extract_image",
            "extract_audio": "extract_audio",
            "extract_pdf": "extract_pdf",
            "embed_text": "embed_text"
        }
    )
    workflow.add_edge("extract_image", "embed_text")
    workflow.add_edge("extract_audio", "embed_text")
    workflow.add_edge("extract_pdf", "embed_text")
    # workflow.add_edge("embed_text", "check_cache")
    # workflow.add_conditional_edges(
    #     "check_cache",
    #     route_after_cache_check,
    #     {
    #         "return_cached": "return_cached",
    #         "generate_new": "generate_new"
    #     }
    # )
    workflow.add_edge("embed_text", "generate_new")
    # workflow.add_edge("return_cached", END)
    workflow.add_edge("generate_new", END)
    return workflow.compile()

from flask import Flask, render_template, request, jsonify
app = Flask(__name__)
chat_workflow = create_chat_workflow()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form.get('message', '')
    session_id = str(request.form.get('session_id'))
    uploaded_file = request.files.get('file')
    file_type = request.form.get('file_type')    
    
    initial_state = {
        "session_id": session_id,
        "user_input": user_message,
        "input_type": "text",
        "file_data": None,
        "extracted_text": "",
        "embedding": None,
        "cached_answer": None,
        "final_answer": "",
        "error_message": None
    }
    
    if uploaded_file and file_type:
        initial_state["file_data"] = {
            "stream": uploaded_file.stream,
            "type": file_type,
            "name": uploaded_file.filename
        }
    if "answer only" in user_message.lower():
        initial_state["user_preference"] = "short"
    else:
        initial_state["user_preference"] = "detailed"

    
    try:
        history = get_last_n_messages(session_id, 10)
        conversation_context = ""
        for idx, (q, a) in enumerate(history):
            conversation_context += f"User: {q}\nBot: {a}\n"

        # Merge history with current user input
        conversation_input = f"{conversation_context}User: {user_message}"
        initial_state["user_input"] = conversation_input
        result = chat_workflow.invoke(initial_state)
        response_text = result.get("final_answer", "Sorry, I couldn't process your request.")
        
        if result.get("error_message"):
            print(f"Workflow error: {result['error_message']}")
        
        return jsonify({"response": response_text})
    except Exception as e:
        print(f"Workflow execution error: {str(e)}")
        return jsonify({"response": "Sorry, there was an error processing your request."})

if __name__ == '__main__':
    from models import clear_database    
    db_path = 'chat_history.db'
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"Removed existing database file: {db_path}")
        except OSError as e:
            print(f"Error removing database file {db_path}: {e}")
    clear_database()
    print("✅ Database initialized")
    app.run(debug=True, use_reloader=False)

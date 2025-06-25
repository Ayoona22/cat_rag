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
    token_count: Optional[int]
    too_large: Optional[bool]
    chat_history: Optional[str]
    chat_summary: Optional[str]
    retrieved_chunks: Optional[list]

SIMILARITY_THRESHOLD = 0.9  
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 🔁 GLOBAL prompt structure for Gemini
BASE_SYSTEM_PROMPT = """
You are a JEE Chemistry assistant.

Instructions:
- Answer only questions related to Chemistry and JEE.
- Prioritize using the study materials provided.
- If the study materials are not relevant, rely on your own knowledge.
- Use precise and helpful explanations. If the user prefers, answer briefly.
- Provide formulas, equations, and step-by-step logic when needed.

Use the following:
1. Study Material (retrieved from memory/vector store):  
{study_material}

2. Summary of the conversation so far:  
{chat_summary}

3. Latest few Q&A turns (most recent 5):  
{chat_history}

4. Current User Question:  
{user_question}

Now answer based on the above.
"""

def check_input_type(state: ChatState) -> ChatState:
    print("Checking input type...")
    if state.get("file_data"):
        file_type = state["file_data"].get("type", "")
        if file_type.startswith('image/'):
            state["input_type"] = "image"
        elif file_type.startswith('audio/'):
            state["input_type"] = "audio"
        else:
            state["input_type"] = "text"
    else:
        state["input_type"] = "text"
    return state

def route_after_input_check(state: ChatState) -> str:
    input_type = state.get("input_type", "text")
    if input_type == "image":
        return "extract_image"
    elif input_type == "audio":
        return "extract_audio"
    else:
        return "embed_text"

def extract_text_from_image(state: ChatState) -> ChatState:
    print("Extracting text from image using OCR...")
    try:
        file_data = state["file_data"]
        image = Image.open(file_data["stream"]).convert('L')  # Convert to grayscale

        # Define fallback configurations
        configs = [r'--oem 3 --psm 6', r'--oem 3 --psm 11']
        extracted_text = ""
        
        for config in configs:
            extracted_text = pytesseract.image_to_string(image, config=config)
            processed_text = preprocess_text(extracted_text)
            if processed_text:
                state["extracted_text"] = f"[Image Content]: {processed_text}"
                break
        else:
            state["extracted_text"] = "[Image Content]: No readable text found."

    except Exception as e:
        state["error_message"] = f"Image processing error: {str(e)}"
        state["extracted_text"] = "[Image Content]: Could not extract text."

    return state

def preprocess_text(text):
    """Clean and preprocess text for better embeddings"""
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s\.\,\?\!\-\(\)]', ' ', text)
    return text

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
        
    except Exception as e:
        state["error_message"] = f"Audio processing error: {str(e)}"
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

def route_after_cache_check(state: ChatState) -> str:
    if state.get("cached_answer"):
        return "return_cached"
    else:
        return "compare_store"
    
def return_cached_answer(state: ChatState) -> ChatState:
    print("Returning cached answer...")
    state["final_answer"] = state["cached_answer"]
    return state

from chromadb import PersistentClient

SIMILARITY_DISTANCE_THRESHOLD = 0.75  
chroma_client = PersistentClient(path="./rag_store")
collection = chroma_client.get_or_create_collection("rag_documents")
def compare_with_vector_store(state: ChatState) -> ChatState:
    print("🔎 Comparing with vector store...")
    try:
        if not state.get("embedding"):
            state["retrieved_chunks"] = []
            return state
        results = collection.query(
            query_embeddings=[state["embedding"]],
            n_results=5,
            include=["documents", "distances"]
        )
        chunks = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        filtered_chunks = [
            chunk for chunk, dist in zip(chunks, distances)
            if dist < SIMILARITY_DISTANCE_THRESHOLD
        ]
        if not filtered_chunks and chunks:
            filtered_chunks = chunks[:2]

        state["retrieved_chunks"] = filtered_chunks

    except Exception as e:
        print(f"❌ Vector store comparison error: {str(e)}")
        state["retrieved_chunks"] = []
        state["error_message"] = f"Vector store comparison error: {str(e)}"

    return state

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

MAX_TOKENS = 30000

def check_tokens(state: ChatState) -> ChatState:
    print("🔢 Counting tokens in context...")

    # Combine the full context
    prompt_text = BASE_SYSTEM_PROMPT.format(
        study_material="\n".join(state.get("retrieved_chunks", [])),
        chat_summary=state.get("chat_summary", ""),
        chat_history=state.get("chat_history", ""),
        user_question=state.get("user_input", "")
    )

    # Tokenize and count
    token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    token_count = len(token_ids)

    print(f"🔍 Token count: {token_count}")

    state["token_count"] = token_count
    state["too_large"] = token_count > MAX_TOKENS

    return state

def route_after_token_check(state: ChatState) -> str:
    return "truncate_context" if state.get("too_large") else "generate_new"

def truncate_context_node(state: ChatState) -> ChatState:
    print("✂️ Truncating context due to token overflow...")

    try:
        # 1. Truncate vector store chunks
        chunks = state.get("retrieved_chunks", [])
        state["retrieved_chunks"] = chunks[:3]  # Keep top 3 most relevant

        # 2. Re-summarize older history to shrink it
        full_summary = state.get("chat_summary", "")
        if full_summary:
            summary_response = model.generate_content(
                f"Summarize briefly again in <100 words:\n{full_summary}"
            )
            state["chat_summary"] = summary_response.text.strip()

        # 3. Optionally drop even older history if needed (e.g. >5K chars)
        history = state.get("chat_history", "")
        if len(history) > 5000:
            lines = history.strip().split("\n")
            state["chat_history"] = "\n".join(lines[-40:])  # last 20 turns max

        print("✅ Truncation complete.")

    except Exception as e:
        print(f"Truncation error: {str(e)}")
        state["error_message"] = f"Truncation error: {str(e)}"

    return state

def generate_answer(state: ChatState) -> ChatState:
    print("Generating answer using Gemini...")

    try:
        if not session_exists(state["session_id"]):
            save_session(state["session_id"], datetime.utcnow())

        study_material = "\n\n".join(state.get("retrieved_chunks", []))
        chat_summary = state.get("chat_summary", "")
        chat_history = state.get("chat_history", "")
        user_question = state.get("user_input", "")

        prompt = BASE_SYSTEM_PROMPT.format(
            study_material=study_material,
            chat_summary=chat_summary,
            chat_history=chat_history,
            user_question=user_question
        )

        response = model.generate_content(prompt)
        answer = response.text.strip()

        save_user_question(
            state["session_id"],
            user_question,
            answer,
            embedder.encode(user_question).tolist()
        )

        state["final_answer"] = answer

    except Exception as e:
        print(f"❌ Error generating answer: {e}")
        state["error_message"] = f"Gemini generation error: {str(e)}"
        state["final_answer"] = "Sorry, I couldn't generate an answer due to an internal error."

    return state


def create_chat_workflow():
    workflow = StateGraph(ChatState)
    workflow.add_node("check_input", check_input_type)
    workflow.add_node("extract_image", extract_text_from_image)
    workflow.add_node("extract_audio", extract_text_from_audio)
    workflow.add_node("embed_text", embed_text)
    workflow.add_node("check_cache", check_cache)
    workflow.add_node("return_cached", return_cached_answer)
    workflow.add_node("compare_store", compare_with_vector_store)
    workflow.add_node("count_tokens", check_tokens)
    workflow.add_node("truncate_context", truncate_context_node)
    workflow.add_node("generate_new", generate_answer)

    workflow.set_entry_point("check_input")    
    workflow.add_conditional_edges(
        "check_input",
        route_after_input_check,
        {
            "extract_image": "extract_image",
            "extract_audio": "extract_audio",
            "embed_text": "embed_text"
        }
    )
    workflow.add_edge("extract_image", "embed_text")
    workflow.add_edge("extract_audio", "embed_text")
    workflow.add_edge("embed_text", "check_cache")
    workflow.add_conditional_edges(
        "check_cache",
        route_after_cache_check,
        {
            "return_cached": "return_cached",
            "compare_store": "compare_store"
        }
    )
    workflow.add_edge("return_cached", END)
    workflow.add_edge("compare_store", "count_tokens")
    workflow.add_conditional_edges(
    "count_tokens",
    route_after_token_check,
    {
        "truncate_context": "truncate_context",  
        "generate_new": "generate_new"
    }
    )
    workflow.add_edge("truncate_context", "count_tokens")
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
        "error_message": None,
        "user_preference": "short" if "answer only" in user_message.lower() else "detailed",  
        "token_count": 0,
        "too_large": False,
        "chat_history": "",
        "chat_summary": "",
        "retrieved_chunks": []
    }

    if uploaded_file and file_type:
        initial_state["file_data"] = {
            "stream": uploaded_file.stream,
            "type": file_type,
            "name": uploaded_file.filename
        }
    try:
        all_history = get_last_n_messages(session_id, 50)
        print(all_history)
        recent_history = all_history[-5:]
        old_history = all_history[:-5]
        chat_history_str = "\n".join([f"User: {q}\nBot: {a}" for q, a in recent_history])
        initial_state["chat_history"] = chat_history_str

        if old_history:
                conversation_so_far = "\n".join([f"User: {q}\nBot: {a}" for q, a in old_history])
                summary_response = model.generate_content(
                    f"Summarize this conversation briefly:\n{conversation_so_far}"
                )
                initial_state["chat_summary"] = summary_response.text.strip()
                print("🟢 Summary of conversation so far:", initial_state["chat_summary"])
        result = chat_workflow.invoke(initial_state)
        response_text = result.get("final_answer", "Sorry, I couldn't generate a response.")
        if result.get("error_message"):
            print(f"⚠️ Error during workflow: {result['error_message']}")

        return jsonify({"response": response_text})

    except Exception as e:
        print(f"❌ Chat route error: {str(e)}")
        return jsonify({"response": "Sorry, an unexpected error occurred."})

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

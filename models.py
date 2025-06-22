from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from json import dumps, loads
from sentence_transformers import util

from sqlalchemy import Float  # At top with other imports

Base = declarative_base()

class ChatSession(Base):
    __tablename__ = 'chat_sessions'
    id = Column(String, primary_key=True)  # use UUID or timestamp as string
    title = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ChatMessage(Base):
    __tablename__ = 'chat_messages'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String, nullable=False)
    role = Column(String(10), nullable=False) # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class ChatSummary(Base):
    __tablename__ = 'chat_summary'

    id = Column(Integer, primary_key=True)
    session_id = Column(String, nullable=False)
    summary = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow)

class UserQuestion(Base):
    __tablename__ = 'user_questions'

    id = Column(Integer, primary_key=True)
    session_id = Column(String, nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    embedding = Column(Text, nullable=False)

# Initialize database
engine = create_engine('sqlite:///chat_history.db', connect_args={'check_same_thread': False})
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def save_session(session_id, created_at):
    session = Session()
    new_session = ChatSession(id=session_id, created_at=created_at)
    session.add(new_session)
    print("🛠️ Saving session:", session_id)
    session.commit()
    session.close()

def session_exists(session_id):
    session = Session()
    exists = session.query(ChatSession).filter_by(id=session_id).first() is not None
    session.close()
    return exists


def save_chat_message(session_id, role, content):
    session = Session()
    new_message = ChatMessage(session_id=session_id, role=role, content=content)
    session.add(new_message)
    session.commit()
    session.close()

def get_chat_history(session_id):
    session = Session()
    history = session.query(ChatMessage).filter_by(session_id=session_id).order_by(ChatMessage.created_at).all()
    session.close()
    return [{ "role": msg.role, "content": msg.content } for msg in history]

def get_summary(session_id):
    session = Session()
    summary = session.query(ChatSummary).filter_by(session_id=session_id).first()
    session.close()
    return summary.summary if summary else ""

def update_summary(session_id, summary_text):
    session = Session()
    try:
        summary = session.query(ChatSummary).filter_by(session_id=session_id).first()
        if summary:
            print("🟡 Updating existing summary for session:", session_id)
            summary.summary = summary_text
        else:
            print("🟢 Creating new summary for session:", session_id)
            summary = ChatSummary(session_id=session_id, summary=summary_text)
            session.add(summary)
        session.commit()
    except Exception as e:
        print("Error updating summary:", e)
    finally:
        session.close()


def save_user_question(session_id,question, answer, embedding):
    session = Session()
    from json import dumps
    entry = UserQuestion(session_id=session_id,question=question, answer=answer, embedding=dumps(embedding))
    session.add(entry)
    session.commit()
    session.close()

def find_similar_question(session_id,question_embedding, threshold):
    from json import loads
    session = Session()
    all_entries = session.query(UserQuestion).all()
    for entry in all_entries:
        if entry.session_id != session_id:
            continue
        else:
            stored_embedding = loads(entry.embedding)
            similarity = util.cos_sim(question_embedding, [stored_embedding])[0][0].item()
            if similarity >= threshold:
                session.close()
                return entry.answer
    session.close()
    return None

    
def clear_database():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine) 
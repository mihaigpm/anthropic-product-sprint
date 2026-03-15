from sqlalchemy import Column, Integer, String, Text, DateTime, func
from database import Base

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)  # Links messages to a specific chat session
    role = Column(String)                    # 'user' or 'assistant' (or 'system')
    content = Column(Text)                   # The actual markdown/text
    created_at = Column(DateTime(timezone=True), server_default=func.now())
from pydantic import BaseModel
from datetime import datetime

class MessageResponse(BaseModel):
    id: int
    session_id: str
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True # This tells Pydantic to read data from SQLAlchemy objects

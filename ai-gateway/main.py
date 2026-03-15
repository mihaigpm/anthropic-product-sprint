import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
import uvicorn

load_dotenv()

CURRENT_MODEL = "claude-sonnet-4-6"

@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not found")
    
    app.state.anthropic_client = AsyncAnthropic(api_key=api_key)
    yield
    await app.state.anthropic_client.close()

app = FastAPI(title="Anthropic Product Sprint - Gateway", lifespan=lifespan)

class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str

class ChatRequest(BaseModel):
    model: str = CURRENT_MODEL
    messages: List[Message]
    max_tokens: Optional[int] = 4096
    temperature: float = Field(default=0.7, ge=0, le=1)

@app.post("/v1/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = await app.state.anthropic_client.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            messages=[msg.model_dump() for msg in request.messages]
        )
        
        return {
            "id": response.id,
            "content": response.content[0].text,
            "usage": response.usage
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

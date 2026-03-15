import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from anthropic import AsyncAnthropic # Use the Async client
from dotenv import load_dotenv
import uvicorn

load_dotenv()

# 1. Manage the Lifespan of the Client
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create the client
    app.state.anthropic_client = AsyncAnthropic()
    yield
    # Shutdown: Cleanly close the client
    await app.state.anthropic_client.close()

# Pass the lifespan to the FastAPI app
app = FastAPI(title="Anthropic Product Sprint - Gateway", lifespan=lifespan)

# 1. Structured Data Models
class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str

class ChatRequest(BaseModel):
    model: str = "claude-3-5-sonnet-20240620"
    messages: List[Message]
    max_tokens: Optional[int] = 1024
    temperature: float = Field(default=0.7, ge=0, le=1)

# 2. Health Check (Standard for Kubernetes Liveness Probes)
@app.get("/health")
def health_check():
    return {"status": "healthy", "cluster": "local-orbstack"}

# 3. The Core Endpoint
@app.post("/v1/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        formatted_messages = [msg.model_dump() for msg in request.messages]
        response = await app.state.anthropic_client.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            messages=formatted_messages
        )

        return {
            "id": response.id,
            "content": response.content[0].text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
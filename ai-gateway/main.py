import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
import json
import asyncio
import uvicorn
import time
from starlette.middleware.base import BaseHTTPMiddleware

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
    system_prompt: str = "You are a helpful, senior-level coding assistant."
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
    
@app.post("/v1/chat/stream")
async def chat_stream(request: ChatRequest):
    start_time = time.time()
    # The event generator handles the persistent connection
    async def event_generator():
        start_time = asyncio.get_event_loop().time() # Use monotonic time
        first_token_received = False
        try:
            async with app.state.anthropic_client.messages.stream(
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system=[
                    {
                        "type": "text",
                        "text": request.system_prompt,
                        "cache_control": {"type": "ephemeral"}
                    }
                ],
                messages=[msg.model_dump() for msg in request.messages]
            ) as stream:
                # Loop through the stream as tokens are generated
                async for text in stream.text_stream:
                    if not first_token_received:
                        end_time = asyncio.get_event_loop().time()
                        ttft = end_time - start_time
                        print(f"METRIC | TTFT: {ttft:.4f}s | Model: {request.model}")
                        first_token_received = True
                    # SSE standard format: "data: <JSON>\n\n"
                    # Sending as JSON allows the client to parse metadata easily
                    yield f"data: {json.dumps({'text': text})}\n\n"
                    
                # Send a final signal so the client knows to close the connection
                yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

class LatencyLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()
        
        # Process the request
        response = await call_next(request)
        
        # Calculate total duration for non-streaming or setup for streaming
        process_time = time.time() - start_time
        
        # Add a custom header so the client/browser can also see the latency
        response.headers["X-Process-Time"] = str(process_time)
        
        print(f"DEBUG: Path: {request.url.path} | Total processing time: {process_time:.4f}s")
        return response

# Register the middlewares
app.add_middleware(LatencyLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, you'd specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

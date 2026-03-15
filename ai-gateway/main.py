import os
import json
import asyncio
import uvicorn
import time
from tools import registry
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from starlette.middleware.base import BaseHTTPMiddleware
from database import engine, Base, get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from models import Message
import models
import uuid
from schemas import MessageResponse

import json

load_dotenv()

CURRENT_MODEL = "claude-sonnet-4-6"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    print("🚀 Starting API Gateway...")
    
    # Initialize the Anthropic Client
    app.state.anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # Create Database Tables (if they don't exist)
    async with engine.begin() as conn:
        print("📦 Ensuring database tables exist...")
        await conn.run_sync(Base.metadata.create_all)
        
    yield # This is where the application runs
    
    # --- Shutdown Logic ---
    print("🛑 Shutting down API Gateway...")
    await app.state.anthropic_client.close()
    await engine.dispose()

# Now define the app using the lifespan manager
app = FastAPI(lifespan=lifespan)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = CURRENT_MODEL
    system_prompt: str = "You are a helpful, senior-level coding assistant."
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 4096
    temperature: float = Field(default=0.7, ge=0, le=1)

@app.post("/v1/chat/agent")
async def agent_endpoint(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    session_id = "test-session-123"

    # 1. Load History
    # Query the database using the SQLAlchemy class directly from the models module
    # This avoids Pydantic trying to parse it as an API type
    result = await db.execute(
        select(models.Message)
        .where(models.Message.session_id == session_id)
        .order_by(models.Message.created_at)
    )
    db_messages = result.scalars().all()
    
    # 2. Format History (Crucial: Handling tool blocks)
    messages = []
    for msg in db_messages:
        # If the content looks like JSON (for tool results), parse it back
        content = msg.content
        if content.startswith('[') or content.startswith('{'):
            try:
                content = json.loads(content)
            except:
                pass
        messages.append({"role": msg.role, "content": content})
    
    # Add new user message
    new_user_content = request.messages[-1].content
    messages.append({"role": "user", "content": new_user_content})
    db.add(Message(session_id=session_id, role="user", content=new_user_content))

    # 3. Agent Loop
    while True:
        try:
            response = await app.state.anthropic_client.messages.create(
                model=request.model,
                max_tokens=2048,
                tools=registry.get_definitions(),
                messages=messages
            )
        except Exception as e:
            print(f"❌ Anthropic API Error: {e}")
            await db.rollback()
            return {"content": f"I hit a snag calling the AI: {str(e)}"}

        if response.stop_reason != "tool_use":
            final_text = response.content[0].text
            break

        # Handle tools
        # We store the assistant's "thinking" blocks in memory for this loop
        messages.append({"role": "assistant", "content": response.content})
        
        tool_tasks = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                task = registry.execute_tool(content_block.name, content_block.input)
                tool_tasks.append((content_block.id, task))

        results = await asyncio.gather(*(t[1] for t in tool_tasks))

        for i, (tool_use_id, _) in enumerate(tool_tasks):
            result_block = {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": str(results[i]),
            }
            messages.append({"role": "user", "content": [result_block]})
        
    # 4. Save Final Answer
    db.add(Message(session_id=session_id, role="assistant", content=final_text))
    await db.commit()

    return {"content": final_text}
    
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

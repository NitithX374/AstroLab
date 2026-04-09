from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from database import db
from routes.auth import get_current_user
from datetime import datetime
from bson import ObjectId
import json
import asyncio
import time
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
router = APIRouter()

class AskRequest(BaseModel):
    prompt: str

# Mock LLM generation simulating a real LLM stream
async def mock_llm_stream(context_messages: list):
    prompt_text = context_messages[-1]['content']
    
    response_tokens = [
        "Thinking", " about", f" '{prompt_text}'", "... ",
        "Based", " on", " the", " context", " provided", ",", " I", " am", " an",
        " AI", " CLI", " assistant", " that", " streams", " tokens", " directly", " to", " your", " browser", " terminal."
    ]
    
    for token in response_tokens:
        await asyncio.sleep(0.05) # Simulate latency
        yield token

async def get_or_create_conversation(user_id: ObjectId):
    conv = await db.conversations.find_one({"user_id": user_id}, sort=[("updated_at", -1)])
    if not conv:
        conv_doc = {
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        result = await db.conversations.insert_one(conv_doc)
        conv = await db.conversations.find_one({"_id": result.inserted_id})
    return conv

@router.post("/ask/stream")
@limiter.limit("5/second")
async def ask_stream(request: Request, ask_req: AskRequest, current_user: dict = Depends(get_current_user)):
    user_id = current_user["_id"]
    conv = await get_or_create_conversation(user_id)
    conv_id = conv["_id"]
    
    user_msg = {
        "conversation_id": conv_id,
        "role": "user",
        "content": ask_req.prompt,
        "timestamp": datetime.utcnow()
    }
    await db.messages.insert_one(user_msg)
    
    await db.conversations.update_one(
        {"_id": conv_id},
        {"$set": {"updated_at": datetime.utcnow()}}
    )

    cursor = db.messages.find({"conversation_id": conv_id}).sort("timestamp", -1).limit(20)
    messages = await cursor.to_list(length=20)
    messages.reverse()
    
    async def event_generator():
        start_time = time.time()
        full_response = ""
        token_count = 0
        try:
            # Wrap the mock generator in an asyncio timeout (e.g. 30 seconds max)
            async with asyncio.timeout(30.0):
                async for token in mock_llm_stream(messages):
                    full_response += token
                    token_count += 1
                    yield f"data: {token}\n\n"
                    
            bot_msg = {
                "conversation_id": conv_id,
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.utcnow()
            }
            await db.messages.insert_one(bot_msg)
            
            latency_ms = (time.time() - start_time) * 1000
            await db.ask_logs.insert_one({
                "user_id": user_id,
                "prompt": ask_req.prompt,
                "response": full_response,
                "tokens_used": token_count,
                "latency_ms": latency_ms,
                "model": "mock-llm-v1",
                "status": "success",
                "created_at": datetime.utcnow()
            })
            
            yield "data: [DONE]\n\n"
            
        except asyncio.TimeoutError:
            yield "data: [ERROR] LLM generation timed out.\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
            latency_ms = (time.time() - start_time) * 1000
            await db.ask_logs.insert_one({
                "user_id": user_id,
                "prompt": ask_req.prompt,
                "response": full_response,
                "tokens_used": token_count,
                "latency_ms": latency_ms,
                "model": "mock-llm-v1",
                "status": "error",
                "created_at": datetime.utcnow()
            })

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/ask")
@limiter.limit("5/second")
async def ask_sync(request: Request, ask_req: AskRequest, current_user: dict = Depends(get_current_user)):
    user_id = current_user["_id"]
    conv = await get_or_create_conversation(user_id)
    conv_id = conv["_id"]
    
    user_msg = {
        "conversation_id": conv_id,
        "role": "user",
        "content": ask_req.prompt,
        "timestamp": datetime.utcnow()
    }
    await db.messages.insert_one(user_msg)
    
    cursor = db.messages.find({"conversation_id": conv_id}).sort("timestamp", -1).limit(20)
    messages = await cursor.to_list(length=20)
    messages.reverse()

    start_time = time.time()
    full_response = ""
    token_count = 0
    try:
        async with asyncio.timeout(30.0):
            async for token in mock_llm_stream(messages):
                full_response += token
                token_count += 1
                
        bot_msg = {
            "conversation_id": conv_id,
            "role": "assistant",
            "content": full_response,
            "timestamp": datetime.utcnow()
        }
        await db.messages.insert_one(bot_msg)
        
        latency_ms = (time.time() - start_time) * 1000
        await db.ask_logs.insert_one({
            "user_id": user_id,
            "prompt": ask_req.prompt,
            "response": full_response,
            "tokens_used": token_count,
            "latency_ms": latency_ms,
            "model": "mock-llm-v1",
            "status": "success",
            "created_at": datetime.utcnow()
        })
        
        return {"response": full_response}
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        await db.ask_logs.insert_one({
            "user_id": user_id,
            "prompt": ask_req.prompt,
            "response": full_response,
            "tokens_used": token_count,
            "latency_ms": latency_ms,
            "model": "mock-llm-v1",
            "status": "error",
            "created_at": datetime.utcnow()
        })
        return {"error": str(e)}

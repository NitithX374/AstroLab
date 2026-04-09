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
import os
from anthropic import AsyncAnthropic
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
router = APIRouter()

class AskRequest(BaseModel):
    prompt: str

# Initialize Anthropic Client
anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "missing-key"))

async def get_real_llm_stream(context_messages: list, user_id: str):
    # Check if API key was provided
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "missing-key":
        yield "[ERROR] Anthropic API Key is missing in Render environment variables."
        return

    from astrolab_session import get_astrolab_session
    
    # 1. Get the actual simulation state from this user's Python engine
    cli = get_astrolab_session(user_id)
    sim_state_summary = "No simulation state available."
    
    if hasattr(cli, 'manager') and cli.manager.state:
        # Create a brief summary of the physics engine for the AI
        bodies = [f"{b.name} ({b.body_type})" for b in cli.manager.state.bodies]
        sim_state_summary = f"Current bodies in simulation: {', '.join(bodies)}. Timestep (dt): {cli.manager.state.dt}s."

    # 2. Inject this as a System Prompt
    system_prompt = (
        "You are the AstroLab AI assistant. You help users with astrophysical simulations. "
        f"CONTEXT: {sim_state_summary} "
        "User is interacting via a Web CLI terminal. Keep responses concise and professional."
    )

    # 3. Stream from Anthropic
    try:
        async with anthropic_client.messages.stream(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system=system_prompt,
            messages=context_messages
        ) as stream:
            async for text in stream.text_stream:
                yield text
    except Exception as e:
        yield f"[ERROR] Anthropic API failed: {str(e)}"

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
    
    # Format messages for Anthropic (remove ObjectIds)
    anthropic_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m["role"] in ("user", "assistant")
    ]
    
    async def event_generator():
        start_time = time.time()
        full_response = ""
        token_count = 0
        try:
            # Wrap the real generator in an asyncio timeout
            async with asyncio.timeout(30.0):
                # We need to pass user_id (string) to get simulation context
                async for token in get_real_llm_stream(anthropic_messages, str(user_id)):
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

    anthropic_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m["role"] in ("user", "assistant")
    ]

    start_time = time.time()
    full_response = ""
    token_count = 0
    try:
        async with asyncio.timeout(30.0):
            async for token in get_real_llm_stream(anthropic_messages, str(user_id)):
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

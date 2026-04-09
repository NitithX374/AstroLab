from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import EventSourceResponse
from pydantic import BaseModel
import sys
import asyncio
import io
from queue import Queue, Empty
from contextlib import redirect_stdout, redirect_stderr
from concurrent.futures import ThreadPoolExecutor

from routes.auth import get_current_user
from astrolab_session import get_astrolab_session, clear_astrolab_session

router = APIRouter()
executor = ThreadPoolExecutor(max_workers=10)

class CommandRequest(BaseModel):
    command: str

class QueueWriter(io.TextIOBase):
    def __init__(self, q: Queue):
        self.q = q
    def write(self, string):
        if string:
            self.q.put(string)
        return len(string)
    def flush(self):
        pass

def execute_command_in_thread(cli, command: str, out_queue: Queue):
    writer = QueueWriter(out_queue)
    # Save the original stdout reference so we don't break the CLI if used elsewhere
    original_stdout = getattr(cli, 'stdout', sys.stdout)
    
    try:
        # Override the bound stdout on the cmd.Cmd instance so internal methods
        # like do_help() log to our custom writer instead of the old sys.stdout
        cli.stdout = writer
        
        with redirect_stdout(writer), redirect_stderr(writer):
            cli.onecmd(command)
    except Exception as e:
        # Catch unexpected crashes
        writer.write(f"\n[Error]: {str(e)}\n")
    finally:
        # Restore the original reference
        cli.stdout = original_stdout
        out_queue.put("[DONE]")

@router.post("/execute/stream")
async def execute_astrolab_command(request: Request, cmd_req: CommandRequest, current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["_id"])
    command = cmd_req.command.strip()
    
    # Handle explicit reset command
    if command.lower() in ("reset session", "clear state"):
        clear_astrolab_session(user_id)
        
        async def reset_stream():
            yield 'data: {"text": "AstroLab session reset and state cleared.\\n"}\n\n'
            yield 'data: {"text": "[DONE]"}\n\n'
        return EventSourceResponse(reset_stream())
        
    cli = get_astrolab_session(user_id)
    out_queue = Queue()
    
    # Fire off the synchronous command execution to a thread pool
    # so we don't block the FastAPI async event loop
    loop = asyncio.get_running_loop()
    loop.run_in_executor(executor, execute_command_in_thread, cli, command, out_queue)
    
    async def sse_generator():
        import json
        while True:
            try:
                # Use a small timeout to allow async loop to breathe
                chunk = out_queue.get_nowait()
                if chunk == "[DONE]":
                    yield 'data: {"text": "[DONE]"}\n\n'
                    break
                else:
                    payload = json.dumps({"text": chunk})
                    yield f"data: {payload}\n\n"
            except Empty:
                await asyncio.sleep(0.05) # Yield control back to loop to allow streaming
                
    return EventSourceResponse(sse_generator())

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

@router.get("/state")
async def get_simulation_state(current_user: dict = Depends(get_current_user)):
    """Return structured simulation state for the right-panel Body Inspector."""
    user_id = str(current_user["_id"])
    cli = get_astrolab_session(user_id)

    bodies = []
    sim_time = 0.0
    sim_step = 0
    sim_dt = 60.0
    bh_config = None

    G = 6.6743e-11
    c = 299792458.0

    if hasattr(cli, 'manager') and cli.manager.state:
        state = cli.manager.state
        sim_time = state.time
        sim_step = getattr(state, 'step', 0)
        sim_dt = state.dt

        for b in state.bodies:
            # Schwarzschild radius: r_s = 2GM/c²
            r_s = (2 * G * b.mass) / (c ** 2)
            bodies.append({
                "name": b.name,
                "type": b.body_type,
                "mass_kg": b.mass,
                "radius_m": b.radius,
                "schwarzschild_radius_m": r_s,
                "position": {"x": b.position.x, "y": b.position.y, "z": b.position.z},
                "velocity": {"x": b.velocity.x, "y": b.velocity.y, "z": b.velocity.z},
            })

    if hasattr(cli, '_bh_config') and cli._bh_config is not None:
        bh = cli._bh_config
        bh_config = {
            "mass_kg": bh.mass_kg,
            "metric": bh.metric_type,
            "spin": bh.spin,
            "schwarzschild_radius_km": bh.schwarzschild_radius_km,
        }

    return {
        "bodies": bodies,
        "sim_time_s": sim_time,
        "sim_step": sim_step,
        "sim_dt_s": sim_dt,
        "bh_config": bh_config,
    }


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

import sys
import os
import asyncio
from datetime import datetime
from typing import Dict, Any

# Ensure astrolab core engine is in the path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from astrolab.cli.parser import AstroLabCLI

# In-memory session store
# Mapping of JWT string or Username -> Session Dict
# {
#    "user_id_123": {
#        "cli": AstroLabCLI(),
#        "last_accessed": datetime.utcnow()
#    }
# }
active_sessions: Dict[str, Dict[str, Any]] = {}

SESSION_TIMEOUT_SECONDS = 3600  # 1 hour idle timeout

def get_astrolab_session(user_id: str) -> AstroLabCLI:
    """Retrieve or create a stateful AstroLabCLI instance for the user."""
    # Ensure Windows path stuff doesn't break CLI when running remotely on Linux
    
    if user_id not in active_sessions:
        # Initialize a new AstroLab CLI instance for this user
        cli = AstroLabCLI()
        # Suppress the default welcome banner since and prevent it from trying to read stdin
        cli.intro = "" 
        
        active_sessions[user_id] = {
            "cli": cli,
            "last_accessed": datetime.utcnow()
        }
    
    active_sessions[user_id]["last_accessed"] = datetime.utcnow()
    return active_sessions[user_id]["cli"]

def clear_astrolab_session(user_id: str) -> None:
    """Explicitly delete a user's session to free memory and reset state."""
    if user_id in active_sessions:
        del active_sessions[user_id]

async def cleanup_idle_sessions():
    """Background task to remove idle sessions and prevent memory leaks."""
    while True:
        try:
            await asyncio.sleep(300) # Check every 5 minutes
            now = datetime.utcnow()
            expired_users = []
            
            for user_id, session_data in active_sessions.items():
                idle_time = (now - session_data["last_accessed"]).total_seconds()
                if idle_time > SESSION_TIMEOUT_SECONDS:
                    expired_users.append(user_id)
            
            for user_id in expired_users:
                print(f"🧹 Clearing idle AstroLab session for user: {user_id}")
                del active_sessions[user_id]
                
        except Exception as e:
            print(f"Error in cleanup_idle_sessions: {e}")

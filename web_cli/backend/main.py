from dotenv import load_dotenv
import os

load_dotenv() # Load variables from .env file

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from database import connect_to_mongo, close_mongo_connection
from routes import auth, cli, engine
from astrolab_session import cleanup_idle_sessions
import asyncio
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_to_mongo()
    
    # Start background cleanup task for inactive Astrolab CLI sessions
    asyncio.create_task(cleanup_idle_sessions())
    
    yield
    await close_mongo_connection()

app = FastAPI(lifespan=lifespan)
# Allowed Origins (handles both local dev and production from env)
env_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
origins = [o.strip() for o in env_origins.split(",")]

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(cli.router, tags=["cli"])
app.include_router(engine.router, tags=["engine"])

# Mount AstroLab root to serve generated HTML visualization files
outputs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
app.mount("/outputs", StaticFiles(directory=outputs_dir), name="outputs")

# Custom Rate Limit Middleware
@app.middleware("http")
async def apply_rate_limit(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)
    return await call_next(request)

# CORSMiddleware MUST be added LAST to be executed FIRST
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
async def root():
    return {"message": "AI Web CLI Platform API Running"}

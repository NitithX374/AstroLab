from dotenv import load_dotenv
import os

load_dotenv() # Load variables from .env file

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from database import connect_to_mongo, close_mongo_connection
from routes import auth, cli
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_to_mongo()
    yield
    await close_mongo_connection()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Add Rate Limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Allowed Origins (handles both local dev and production from env)
env_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
origins = [o.strip() for o in env_origins.split(",")]

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(cli.router, tags=["cli"])

@app.middleware("http")
async def apply_rate_limit(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)
    return await call_next(request)



@app.get("/")
async def root():
    return {"message": "AI Web CLI Platform API Running"}

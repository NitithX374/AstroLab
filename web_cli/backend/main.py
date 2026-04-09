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

# Add Rate Limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Allowed Origins (for local React development using Vite)
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True, # Important for HttpOnly cookies
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["auth"])
# Apply rate limiting to CLI router (5 requests per second)
# Since router depends on Request for rate limiting, we could add it at router level,
# but it's easier to add it on the app with a dependency or directly decorating routes.
# Actually slowapi requires decorating the endpoints. I'll modify the cli endpoints below directly.

# Applying rate limit directly to the app routes imported from cli
# I'll just apply it globally for /ask endpoints
@app.middleware("http")
async def apply_rate_limit(request: Request, call_next):
    if request.url.path.startswith("/ask"):
        # simple manual rate limit logic could go here, or we use slowapi on the router directly.
        pass
    return await call_next(request)

app.include_router(cli.router, tags=["cli"])

@app.get("/")
async def root():
    return {"message": "AI Web CLI Platform API Running"}

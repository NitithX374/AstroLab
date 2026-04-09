from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DATABASE_NAME = "ai_cli_db"

# We initialize these globally so they are available immediately
client = AsyncIOMotorClient(MONGO_URL)
db = client[DATABASE_NAME]

async def connect_to_mongo():
    # Verify the connection
    try:
        await client.admin.command('ping')
        print(f"Connected to MongoDB Atlas successfully")
    except Exception as e:
        print(f"Could not connect to MongoDB: {e}")

async def close_mongo_connection():
    client.close()

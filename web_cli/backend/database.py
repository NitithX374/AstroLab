from motor.motor_asyncio import AsyncIOMotorClient
import os

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DATABASE_NAME = "ai_cli_db"

client = None
db = None

async def connect_to_mongo():
    global client, db
    client = AsyncIOMotorClient(MONGO_URL)
    db = client[DATABASE_NAME]
    print(f"Connected to MongoDB at {MONGO_URL}")

async def close_mongo_connection():
    global client
    if client:
        client.close()
        print("Closed MongoDB connection")

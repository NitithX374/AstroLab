from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from passlib.context import CryptContext
from pydantic import BaseModel
from datetime import datetime, timedelta
import jwt
from typing import Optional
from database import db

# JWT Configuration
SECRET_KEY = "your-very-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 * 24 * 60 # 30 days

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
router = APIRouter()

class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        # Assuming scheme "Bearer <token>" in cookie or just "<token>"
        if token.startswith("Bearer "):
            token = token.split(" ")[1]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        user = await db.users.find_one({"username": username})
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        return user
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")


@router.post("/register")
async def register(user: UserCreate):
    existing_user = await db.users.find_one({"username": user.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    user_doc = {
        "username": user.username,
        "password_hash": hashed_password,
        "created_at": datetime.utcnow()
    }
    
    result = await db.users.insert_one(user_doc)
    return {"message": "User created successfully", "id": str(result.inserted_id)}


@router.post("/login")
async def login(response: Response, user: UserLogin):
    user_in_db = await db.users.find_one({"username": user.username})
    if not user_in_db or not verify_password(user.password, user_in_db["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_in_db["username"]}, expires_delta=access_token_expires
    )
    
    # Set HttpOnly cookie
    response.set_cookie(
        key="access_token",
        value=f"Bearer {access_token}",
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        expires=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
        secure=False # Set to True in production with HTTPS
    )
    return {"message": "Login successful"}

@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie("access_token")
    return {"message": "Logged out successfully"}

@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return {
        "username": current_user["username"],
        "id": str(current_user["_id"])
    }

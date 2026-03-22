from fastapi import Request, HTTPException, Depends, status
from fastapi.security import APIKeyHeader
import hashlib

API_KEYS_DB = {}

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_password_hash(password: str) -> str:
    """Hash password using SHA256 (replace with bcrypt in production)"""
    return hashlib.sha256(password.encode()).hexdigest()

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key is None:
        raise HTTPException(status_code=401, detail="API key missing")
    if api_key in API_KEYS_DB:
        return api_key
    raise HTTPException(status_code=401, detail="Invalid API Key")


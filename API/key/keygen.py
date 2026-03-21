import secrets
from .auth import API_KEYS_DB

def generate_api_key() -> str:
    """Generate secure API key and store in DB"""
    new_key = secrets.token_urlsafe(32)
    API_KEYS_DB[new_key] = {"created": "now", "status": "active"}
    return new_key


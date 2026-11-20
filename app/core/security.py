import os
import secrets
import warnings
from datetime import datetime, timedelta
from typing import Optional
import jwt
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext

class SecurityHandler:
    security = HTTPBearer()
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    # Read the signing key from the environment. If it is not set, use a random
    # per-process key so tokens can never be forged with a key published in the repo
    # (tokens then stop working when the server restarts).
    SECRET_KEY = os.environ.get("JWT_SECRET_KEY") or secrets.token_urlsafe(64)
    if "JWT_SECRET_KEY" not in os.environ:
        warnings.warn("JWT_SECRET_KEY is not set; using a random key for this process.")
    ALGORITHM = "HS256"
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        return SecurityHandler.pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return SecurityHandler.pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SecurityHandler.SECRET_KEY, algorithm=SecurityHandler.ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
        try:
            payload = jwt.decode(credentials.credentials, SecurityHandler.SECRET_KEY, algorithms=[SecurityHandler.ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Could not validate credentials") 
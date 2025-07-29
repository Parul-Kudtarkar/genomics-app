import os
import json
import logging
from typing import Optional, Dict, Any
from functools import wraps
from urllib.request import urlopen
from jose import jwt
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

# Auth0 configuration
AUTH0_DOMAIN = os.getenv('AUTH0_DOMAIN', 'your-domain.auth0.com')
AUTH0_AUDIENCE = os.getenv('AUTH0_AUDIENCE', 'https://your-api-identifier')
AUTH0_ISSUER = f'https://{AUTH0_DOMAIN}/'

# JWKS (JSON Web Key Set) for token validation
jsonurl = urlopen(f'https://{AUTH0_DOMAIN}/.well-known/jwks.json')
jwks = json.loads(jsonurl.read())

# Security scheme
security = HTTPBearer()

def get_token_auth_header(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Obtains the Access Token from the Authorization Header"""
    return credentials.credentials

def requires_auth(func):
    """Decorator to require authentication for endpoints"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        token = kwargs.get('token')
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization token required"
            )
        return await func(*args, **kwargs)
    return wrapper

def requires_scope(required_scope: str):
    """Decorator to require specific scope/permission"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            token = kwargs.get('token')
            if not token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authorization token required"
                )
            
            # Check if token has required scope
            if not has_scope(token, required_scope):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Required scope: {required_scope}"
                )
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def verify_decode_jwt(token: str) -> Dict[str, Any]:
    """Verifies and decodes the JWT token"""
    try:
        # Get the unverified header
        unverified_header = jwt.get_unverified_header(token)
        
        # Get the RSA key
        rsa_key = {}
        for key in jwks["keys"]:
            if key["kid"] == unverified_header["kid"]:
                rsa_key = {
                    "kty": key["kty"],
                    "kid": key["kid"],
                    "use": key["use"],
                    "n": key["n"],
                    "e": key["e"]
                }
        
        if rsa_key:
            try:
                # Use the key to validate the token
                payload = jwt.decode(
                    token,
                    rsa_key,
                    algorithms=["RS256"],
                    audience=AUTH0_AUDIENCE,
                    issuer=AUTH0_ISSUER
                )
                return payload
            except jwt.ExpiredSignatureError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired"
                )
            except jwt.JWTClaimsError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid claims"
                )
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token"
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unable to find appropriate key"
            )
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

def has_scope(token: str, required_scope: str) -> bool:
    """Check if token has required scope"""
    try:
        payload = verify_decode_jwt(token)
        token_scopes = payload.get('scope', '').split()
        return required_scope in token_scopes
    except Exception:
        return False

def get_current_user(token: str = Depends(get_token_auth_header)) -> Dict[str, Any]:
    """Get current authenticated user from token"""
    payload = verify_decode_jwt(token)
    return {
        'sub': payload.get('sub'),
        'email': payload.get('email'),
        'name': payload.get('name'),
        'picture': payload.get('picture'),
        'permissions': payload.get('permissions', []),
        'scope': payload.get('scope', '')
    }

def get_user_permissions(token: str = Depends(get_token_auth_header)) -> list:
    """Get user permissions from token"""
    payload = verify_decode_jwt(token)
    return payload.get('permissions', [])

def require_permission(permission: str):
    """Decorator to require specific permission"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            token = kwargs.get('token')
            if not token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authorization token required"
                )
            
            user_permissions = get_user_permissions(token)
            if permission not in user_permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Required permission: {permission}"
                )
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Rate limiting per user
user_request_counts = {}

def rate_limit_per_user(user_id: str, max_requests: int = 100, window_minutes: int = 60):
    """Simple rate limiting per user"""
    import time
    current_time = time.time()
    window_start = current_time - (window_minutes * 60)
    
    if user_id not in user_request_counts:
        user_request_counts[user_id] = []
    
    # Clean old requests
    user_request_counts[user_id] = [
        req_time for req_time in user_request_counts[user_id] 
        if req_time > window_start
    ]
    
    # Check if user has exceeded limit
    if len(user_request_counts[user_id]) >= max_requests:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    # Add current request
    user_request_counts[user_id].append(current_time) 
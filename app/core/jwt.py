from jose import jwt
from datetime import datetime, timedelta
from app.core.config import settings  # .env에서 PRIVATE_KEY 불러옴

with open(settings.PRIVATE_KEY_PATH, "r") as f:
    PRIVATE_KEY = f.read()

with open(settings.PUBLIC_KEY_PATH, "r") as f:
    PUBLIC_KEY = f.read()
    
def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({
        "exp": expire, 
        "type": "access" # 접근 토큰 
    })

    return jwt.encode(
        to_encode,
        PRIVATE_KEY,
        algorithm="RS256"
    )

# def create_refresh_token(data: dict) -> str:
#     to_encode = data.copy()
#     expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
#     to_encode.update({
#         "exp": expire,
#         "type": "refresh" # 리프레시 토큰 
#     })
#     return jwt.encode(to_encode, _PRIVATE_KEY, algorithm="RS256")

def verify_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(
            token, 
            PUBLIC_KEY, 
            algorithms="RS256"
        )
        return payload
    except Exception as e:
        raise e

# def verify_refresh_token(token: str) -> dict:
#     try:
#         payload = jwt.decode(
#             token,
#             _PUBLIC_KEY,
#             algorithms=["RS256"],
#             options={"require": ["exp", "type"]}
#         )
#     except jwt.PyJWTError as e:
#         raise ValueError(f"접근 불가한 refresh token: {e}")
#     if payload.get("type") != "refresh":
#         raise ValueError("refresh token이 아님")
#     return payload
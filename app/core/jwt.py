from jose import jwt
from datetime import datetime, timedelta
from app.core.config import settings  # .env에서 PRIVATE_KEY 불러옴

with open(settings.PRIVATE_KEY_PATH, "r") as f:
    PRIVATE_KEY = f.read()

with open(settings.PUBLIC_KEY_PATH, "r") as f:
    PUBLIC_KEY = f.read()
    
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})

    return jwt.encode(
        to_encode,
        PRIVATE_KEY,
        algorithm="RS256"
    )

def verify_access_token(token: str):
    try:
        payload = jwt.decode(token, PUBLIC_KEY, algorithms=["RS256"])
        return payload
    except Exception as e:
        raise e

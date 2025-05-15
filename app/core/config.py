from pydantic import BaseSettings
from pathlib import Path

# with open("/home/dragon/ChatStory/keys/private_key.pem", "r") as f:
#     PRIVATE_KEY = f.read()

# with open("/home/dragon/ChatStory/keys/public_key.pem", "r") as f:
#     PUBLIC_KEY = f.read()
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    PRIVATE_KEY_PATH: str = str(BASE_DIR / "private.pem")
    PUBLIC_KEY_PATH: str = str(BASE_DIR / "public.pem")
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    MONGO_URL: str
    DB_NAME: str

    # OPENAI_API_KEY: str = ""

    class Config:
        env_file = str(Path(__file__).resolve().parents[2] / ".env")
        # env_file = ".env"
# 환경 설정 인스턴스
settings = Settings()

# with open(Path(settings.PRIVATE_KEY_PATH), "r") as f:
#     settings.PRIVATE_KEY_PATH = f.read()

# with open(Path(settings.PUBLIC_KEY_PATH), "r") as f:
#     settings.PUBLIC_KEY_PATH = f.read()
    

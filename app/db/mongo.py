from pymongo import MongoClient
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME", "ChatStorysUser")  

client = MongoClient(MONGO_URL)
db = client[DB_NAME]

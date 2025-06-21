from pymongo import MongoClient
from dotenv import load_dotenv
import os

MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME", "ChatStorysUser")

client = MongoClient(MONGO_URL)
db = client[DB_NAME]

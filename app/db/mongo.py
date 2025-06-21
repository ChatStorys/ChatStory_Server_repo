from pymongo import MongoClient
import os

MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME", "ChatStorysUser")

client = MongoClient(MONGO_URL)
db = client[DB_NAME]

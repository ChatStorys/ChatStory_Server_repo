from pymongo import MongoClient
import os

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "chatstory")

client = MongoClient(MONGO_URL)
db = client[DB_NAME]

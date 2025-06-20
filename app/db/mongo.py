from pymongo import MongoClient
import os

MONGO_URL = os.getenv("MONGO_URL", "mongodb+srv://DBadmin:Fxz6FChRFep4NCvb@chatstorys.adsotkt.mongodb.net/?retryWrites=true&w=majority&appName=ChatStorys")
DB_NAME = os.getenv("DB_NAME", "ChatStorysUser")

client = MongoClient(MONGO_URL)
db = client[DB_NAME]

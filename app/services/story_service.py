from pymongo import MongoClient
from datetime import datetime
import os

mongo_url = os.getenv("MONGO_URL")
client = MongoClient(mongo_url)
db = client[os.getenv("DB_NAME", "chatstory")]

def create_story(story, token):
    user_id = token["sub"]
    # 소설 생성
    result = db.stories.insert_one({
        "title": story.title,
        "genre": story.genre,
        "author": user_id,
        "created_at": datetime.now()
    })
    return str(result.inserted_id)
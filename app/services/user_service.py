from passlib.context import CryptContext
from pymongo import MongoClient
import os

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

mongo_url = os.getenv("MONGO_URL")
client = MongoClient(mongo_url)
db = client[os.getenv("DB_NAME", "chatstory")]


def create_user(user_id: str, name: str, password: str):
    if db.users.find_one({"user_id": user_id}):
        raise ValueError("이미 존재하는 ID입니디!")

    hashed_pw = pwd_context.hash(password)

    db.users.insert_one({
        "user_id": user_id,
        "name": name,
        "password": hashed_pw # 해시값이 저장된다
    })

    return {"success": True, "message": "회원가입이 완료되었습니다!"}

client = MongoClient(os.getenv("MONGO_URL"))
db = client[os.getenv("DB_NAME", "chatstory")]

def verify_password(plain_password: str, hashed_password: str) -> bool:
    # print("입력된 비밀번호:", plain_password)
    # print("DB에 저장된 해시:", hashed_password)
    
    try:
        result = pwd_context.verify(plain_password, hashed_password)
        print("비교 결과:", result)
        return result
    except Exception as e:
        print("에러 발생:", e)
        return False
    # return pwd_context.verify(plain_password, hashed_password)

def get_user_by_id(user_id: str):
    return db.users.find_one({"user_id": user_id})
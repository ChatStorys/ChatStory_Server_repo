from pydantic import BaseModel, Field
from typing import List, Optional


class ChatMessageRequest(BaseModel):
    user_id: str
    
class ChatMessageResponse(BaseModel):
    message: str


# AI에서 소설을 생성 후 DB에 소설 내용 저장, 백엔드로 결과 전송
class StorySaveRequest(BaseModel):
    user_id: str
    book_id: str
    prompt: str # AI가 생성한 결과

# 저장 성공 여부 
class StorySaveResponse(BaseModel):
    message: str
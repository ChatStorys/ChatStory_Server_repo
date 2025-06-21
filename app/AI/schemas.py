from pydantic import BaseModel, Field
from typing import List, Optional


class ChatMessageRequest(BaseModel):
    user_id: str
    user_message: str
    book_id: str
    
    
class ChatMessageResponse(BaseModel):
    status: str
    code: int
    message: str
    prompt: str
    

class MusicItem(BaseModel):
    title: str
    artist: str
    
class ChapterEndAIRequest(BaseModel): # 챕터 끝 request!
    user_id: str
    book_id: str


class ChapterEndAIResponse(BaseModel): # 챕터 끝 response!
    status: str
    code: int
    summary: str
    recommanded_music: List[MusicItem]
    
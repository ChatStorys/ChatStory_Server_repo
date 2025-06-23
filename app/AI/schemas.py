from pydantic import BaseModel, Field
from typing import List, Optional


# class ChatMessageRequest(BaseModel):
#     user_id: str
#     user_message: str
#     book_id: str
    
    
# class ChatMessageResponse(BaseModel):
#     status: str
#     code: int
#     message: str
#     prompt: str
    

class MusicItem(BaseModel):
    title: str
    artist: str
        
class ChapterContent(BaseModel):
    chapter_num: int
    content: str                     # 사용자+AI 대화 합본
    recommended_music: Optional[MusicItem] = None
    
class ChapterEndAIRequest(BaseModel): # 챕터 끝 request!
    book_id: str


class ChapterEndAIResponse(BaseModel): # 챕터 끝 response!
    status: str
    code: int
    summary: str
    recommended_music: List[MusicItem]

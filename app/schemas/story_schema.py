from pydantic import BaseModel, Field
from typing import Literal, List, Optional
from datetime import datetime

class StoryCreateRequest(BaseModel): # 소설 새로 생성 시 
    title: str = Field(..., min_length=1, max_length=100)
    category: Literal["동화", "연애", "액션", "판타지", "무협", "스릴러", "추리"]

class StoryCreateResponse(BaseModel):
    status: str
    code: int
    book_id: str
    message: str
    

class FrontChatRequest(BaseModel): # 프론트에서 전달 받을 message
    book_id: str # 소설의 ID (예: "book001")
    prompt: str


class FrontChatResponse(BaseModel): # 프론트에 보낼 message
    status: str
    code: int
    message: str
    prompt: str

# class ChatSendRequest(BaseModel): # 벡->AI로 채팅 요청 함수
#     book_id: str
#     prompt: str

# class ChatSendResponse(BaseModel): # AI로부터 응답받는 함수
#     status: str
#     code: int
#     message: str
#     prompt: str

# class MusicItem(BaseModel):
#     title: str
#     artist: str
    
# class ChapterEndAIRequest(BaseModel): # 챕터 끝 request!
#     book_id: str


# class ChapterEndAIResponse(BaseModel): # 챕터 끝 response!
#     status: str
#     code: int
#     message: str
#     summary: str
#     recommanded_music: List[MusicItem]
    
# 소설 끝내버리기
class FinishStoryRequest(BaseModel):
    book_id: str

class FinishStoryResponse(BaseModel):
    status: str
    code: int
    message: str

class ArchiveItemResponse(BaseModel):
    book_id: str = Field(..., alias="bookId")
    title: str
    # genre: str
    # user_id: str
    # workingFlag: bool
    created_at: datetime = Field(..., alias="createdAt")

    class Config:
        allow_population_by_field_name = True
        
class StoryContentResponse(BaseModel):
    book_id: str = Field(..., alias="bookId")
    title: str
    chapters: List[str]
    created_at: datetime = Field(..., alias="createdAt")

    class Config:
        allow_population_by_field_name = True

class DeleteResponse(BaseModel):
    status: str
    code: int
    book_id: str
    message: str
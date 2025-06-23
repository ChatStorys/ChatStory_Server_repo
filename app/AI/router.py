from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.AI.schemas import (
    ChatMessageRequest, ChatMessageResponse,
    ChapterEndAIRequest, ChapterEndAIResponse
)
from app.AI.client import send_message_to_ai_server, send_chapter_end_to_ai
import httpx
import os
from dotenv import load_dotenv
from typing import List, Optional, Dict
from app.AI.main import handle_chapter_summary_with_music, handle_story_continue

ai_router = APIRouter()
# processor = NovelProcessor()

@ai_router.post("/story/continue")
def continue_story(request: ChatMessageRequest):
    """소설 계속 쓰기 엔드포인트"""
    try:
        # AI 모델을 사용하여 소설 생성
        result = handle_story_continue(
            user_id=request.user_id,
            user_message=request.user_message,
            book_id=request.book_id
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@ai_router.post("/story/chapter/summary_with_music", response_model=ChapterEndAIResponse)
def generate_chapter_summary(request: ChapterEndAIRequest):
    """챕터 요약 및 음악 추천 엔드포인트"""
    try:
        # 챕터 요약 및 음악 추천 생성
        result = handle_chapter_summary_with_music(
            user_id=request.user_id,
            book_id=request.book_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return result


# @router.post("/save", 
#         response_model=ChatMessageResponse,
#         summary="사용자 메시지를 AI 서버로 전달, AI는 내용 생성하고 사용자 내용+ai 생성 내용 저장"
# )
# def chat_story_write(
#     req: ChatMessageRequest,
#     user_id: str = Depends(get_current_user),
# ):
#     try:
#         result = send_message_to_ai_server(req)
#         return result
#     except Exception as e:
#         print("send_message_to_ai_server 에러:", e)
#         raise HTTPException(
#             status_code=status.HTTP_502_BAD_GATEWAY,
#             detail=f"AI 처리 중 오류가 발생했습니다: {e}"
#         )

# # 챕터 끝내기
# @router.post(
#     "/chapter/end",
#     response_model=ChapterEndAIResponse,
#     summary="챕터 종료 요청을 AI 서버로 전달"
# )
# def chapter_end(
#     req: ChapterEndAIRequest,
#     user_id: str = Depends(get_current_user),
# ):
#     """
#     챕터 종료 시 AI 서버에 요약 및 음악 추천 요청을 포워딩합니다.
#     서버에서는 별도 저장을 수행하지 않습니다.
#     """
#     try:
#         return send_chapter_end_to_ai(req)
#     except Exception as e:
#         print("send_chapter_end_to_ai 에러:", e)
#         raise HTTPException(
#             status_code=status.HTTP_502_BAD_GATEWAY,
#             detail=f"AI 처리 중 오류가 발생했습니다. {e}"
#         )

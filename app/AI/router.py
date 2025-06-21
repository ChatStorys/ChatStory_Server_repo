from fastapi import APIRouter, HTTPException
from app.AI.schemas import ChatMessageRequest, ChatMessageResponse
from app.AI.client import send_message_to_ai_server

router = APIRouter()

@router.post("/save", response_model=ChatMessageResponse)
async def chat_story_write(request: ChatMessageRequest):
    try:
        result = await send_message_to_ai_server(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 통신 실패: {str(e)}")

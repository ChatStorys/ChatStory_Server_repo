from fastapi import APIRouter, HTTPException
from app.AI.schemas import ChatMessageRequest, ChatMessageResponse
from app.AI.client import send_message_to_ai_server

# router = APIRouter()

# @router.post("/save", response_model=ChatMessageResponse)
# async def chat_story_write(request: ChatMessageRequest):
#     try:
#         # 여기 아래 코드에서 서버 500 에러가 남
#         # 저 함수를 호출하는데 AI서버 URL이 등록이 안되어있어서 그럴 가능성이 높음
#         # 이 SEND MESSAGE TO AI SERVER 함수는 CLIENT.PY에 있음음
#         result = await send_message_to_ai_server(request)
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail="서버 통신 실패")

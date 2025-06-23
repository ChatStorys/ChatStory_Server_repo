import os
from fastapi import FastAPI
from fastapi import Request
from app.api.v1.routers import auth, story
from app.AI import router
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
# from app.AI.schemas import (
#     # ChatMessageRequest, ChatMessageResponse,
#     ChapterEndAIRequest, ChapterEndAIResponse
# )

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "FRONTEND_ORIGIN")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Authorization"],
)

app.include_router(auth.router, prefix="/auth", tags=["Auth"])
app.include_router(router.ai_router, prefix="/ai", tags=["AI"])
app.include_router(story.router, prefix="/story", tags=["Stories"])

# app.include_router(ai_router.router, prefix="/ai", tags=["AI ChatStory"])

# 루트 엔드포인트
# 1. 서버가 잘 작동 중인지 브라우저에서 확인 가능
# 2. 나중에 버전 정보나 상태 코드 제공에도 활용 가능
# 3. 실제 서비스 오픈 시엔 메인 페이지용으로 바꾸기도 함
@app.get("/")
def root():
    return {"message": "Welcome to ChatStory API!"}

@app.api_route("/{path:path}", methods=["CONNECT"])
async def block_connect(request: Request):
    return PlainTextResponse("CONNECT method not allowed", status_code=405)


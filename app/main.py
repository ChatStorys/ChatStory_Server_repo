from fastapi import FastAPI
from app.api.v1.routers import auth, story

app = FastAPI()

app.include_router(auth.router, prefix="/auth", tags=["Auth"])

app.include_router(story.router, prefix="/stories", tags=["Stories"])
# 루트 엔드포인트
# 1. 서버가 잘 작동 중인지 브라우저에서 확인 가능
# 2. 나중에 버전 정보나 상태 코드 제공에도 활용 가능
# 3. 실제 서비스 오픈 시엔 메인 페이지용으로 바꾸기도 함함
@app.get("/")
def root():
    return {"message": "Welcome to ChatStory API!"}

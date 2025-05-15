from fastapi import APIRouter, HTTPException, Response, Depends, Request
from app.schemas.story_schema import StoryCreate
from app.core.jwt import verify_access_token

router = APIRouter(tags=["Story"])

@router.post("/stories/create")
async def create_story(story: StoryCreate, request: Request):
    # 쿠키에서 토큰 가져오기
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="토큰이 누락되었습니다.")

    # Bearer 제거
    if token.startswith("Bearer "):
        token = token[7:]
    
    # 토큰 검증
    try:
        payload = verify_access_token(token)
        user_id = payload.get("sub")
    except Exception as e:
        raise HTTPException(status_code=401, detail="토큰이 유효하지 않습니다.")

    # 제목과 장르 저장 (예시)
    # 실제로는 DB에 저장해야 함
    return {"message": f"'{story.title}' ({story.genre}) 소설이 생성되었습니다.", "user_id": user_id}

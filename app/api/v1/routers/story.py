from fastapi import APIRouter, HTTPException, status, Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.schemas.story_schema import (
    StoryCreateRequest, StoryCreateResponse,
    FrontChatRequest, FrontChatResponse,
    FinishStoryRequest, FinishStoryResponse, 
    ArchiveItemResponse, StoryContentResponse, DeleteResponse
)
from app.services.story_service import (
    get_creating_story, create_story,
    finish_story,
    list_archived_stories,
    get_story_content, delete_story
)
from app.AI.main import handle_story_continue
from app.core.jwt import verify_access_token
from typing import List
# APIRouter 인스턴스 생성
router = APIRouter()

# HTTP Bearer 토큰 인증 스키마 정의
# auto_error=False 로 설정하면 검증 실패 시 직접 예외 처리 가능
bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(bearer_scheme)
) -> str:
    """
    1. Swagger UI 또는 클라이언트 요청의 Authorization 헤더에서 Bearer 토큰을 파싱
    2. 토큰이 없거나 Bearer 스킴이 아니면 401 반환
    3. 토큰을 검증하여 payload 추출 후, sub(claim)로 user_id 반환
    4. 유효하지 않으면 403 반환
    """
    # 1) creds가 없거나 스킴이 bearer가 아니면 인증 실패
    if not creds or creds.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer 토큰이 필요합니다."
        )
    try:
        # 2) 토큰 검증 및 payload 획득
        payload = verify_access_token(creds.credentials)
        user_id = payload.get("sub")
        # sub(claim)가 없으면 예외 발생
        if not user_id:
            raise ValueError("sub(claim)이 없습니다.")
        return user_id  # 유효한 user_id 반환
    except Exception:
        # 토큰이 유효하지 않거나 만료된 경우
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="유효하지 않은 또는 만료된 토큰입니다."
        )

# 소설 생성
@router.post(
    "/create",
    response_model=StoryCreateResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(bearer_scheme)],  # OpenAPI에 보안 스키마 추가
    summary="새 소설 생성 또는 이어쓰기 모드 진입"
)
async def create_story_router(
    body: StoryCreateRequest,
    user_id: str = Depends(get_current_user),  # 인증된 user_id를 주입
):
    """
    1) 작성 중인 소설이 있으면 이어쓰기 모드 진입
    2) 없으면 새 소설 생성 후 그 ID를 반환
    """
    # 이어쓰기 로직: workingFlag=True인 문서 조회
    existing = get_creating_story(user_id)
    if existing:
        return StoryCreateResponse(
            status="success",
            code=200,
            book_id=str(existing.get("bookId")),  # 기존 소설 ID 반환
            message="작성 중인 소설이 있어, 이어쓰기 모드로 진입합니다."
        )

    # 새 소설 생성
    try:
        new_id = create_story(user_id, body)
        return StoryCreateResponse(
            status="success",
            code=201,
            book_id=new_id,  # 생성된 신규 book_id
            message="새 소설 작성을 시작합니다."
        )
    except Exception as e:
        # 내부 로직 에러 시 서버 로그에 출력 후 500 반환
        print("create_story 에러:", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="서버 오류로 소설 생성에 실패했습니다."
        )
        
@router.post(
    "/send",
    response_model=FrontChatResponse,
    status_code=status.HTTP_200_OK,
    summary="프론트 -> AI 채팅 중계",
)
async def front_chat(
    req: FrontChatRequest,
    current_user: str = Depends(get_current_user),
):
    try:
        result = handle_story_continue(
            user_id=current_user,
            prompt=req.prompt,
            book_id=req.book_id
        )
        return result
    
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"{e}")

# 소설 끝내기
@router.post(
    "/finish", 
    response_model=FinishStoryResponse,
    summary="소설 완료 처리"
)
async def finish_story_router(
    book_id: str = Query(..., description="완료할 소설의 ID"),
    user_id: str = Depends(get_current_user)
):
    """
    workingFlag를 False로 변경하여 소설을 완료 처리합니다.
    """
    try:
        success = finish_story(user_id, book_id)
        if not success:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail="해당 소설을 찾을 수 없습니다.")
        return FinishStoryResponse(
            status="success", code=200,
            message="소설 작성 완료 처리되었습니다."
        )
    except HTTPException:
        raise
    except Exception as e:
        print("finish_story 에러:", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="소설 완료 처리에 실패했습니다.")

# 5. 아카이브 목록 조회
@router.get(
    "/archive",
    response_model=List[ArchiveItemResponse],
    summary="아카이브 목록 조회"
)
async def get_archive_list(
    user_id: str = Depends(get_current_user)
):
    """
    완료된 소설 목록을 반환합니다 (workingFlag=False).
    """
    return list_archived_stories(user_id)

# 6. 아카이브 소설 내용 조회
@router.get(
    "/archive/{book_id}",
    response_model=StoryContentResponse,
    summary="아카이브 소설 내용 조회"
)
async def get_archive_story(
    book_id: str,
    user_id: str = Depends(get_current_user)
):
    story = get_story_content(user_id, book_id)
    if not story:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            detail="해당 소설을 찾을 수 없습니다.")
    return story

# 7. 아카이브 소설 삭제
@router.delete(
    "/archive/{book_id}",
    response_model=DeleteResponse,
    summary="아카이브 소설 삭제"
)
async def delete_archive_story(
    book_id: str,
    user_id: str = Depends(get_current_user)
):
    success = delete_story(user_id, book_id)
    if not success:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            detail="해당 소설을 찾을 수 없습니다.")
    return DeleteResponse(status="success", code=200, book_id=book_id,
                          message="소설이 삭제되었습니다.")

from fastapi import APIRouter, HTTPException, BackgroundTasks, Response, Depends
# 회원 가입
#<<<<<<< HEAD
from app.schemas.user_schema import UserCreate, UserCreateResponse
# from app.redis.queue import queue
#>>>>>>> main
from app.services.user_service import create_user
# 로그인
from app.schemas.user_schema import UserLoginRequest, UserLoginResponse
from app.services.user_service import get_user_by_id, verify_password
from app.core.jwt import create_access_token
from passlib.context import CryptContext

router = APIRouter()
# bcrypt로 비밀번호 해싱 
pwd_context = CryptContext(schemes=['bcrypt'], deprecated = 'auto')

@router.post("/register", response_model=UserCreateResponse)
async def register(user: UserCreate):
    # 1. 이미 존재하는 ID 체크
    if get_user_by_id(user.user_id):
        raise HTTPException(status_code=403, detail="이미 존재하는 ID입니다.")
#<<<<<<< HEAD
#=======

    # 2. 바로 회원가입 처리 (동기/비동기)
    try:
        create_user(user.user_id, user.name, user.password)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"회원 가입 실패: {str(e)}")

    # 3. 성공 응답
    return UserCreateResponse(
        status="success",
        code=200,
        user_id=user.user_id,
        message="회원 가입이 완료되었습니다."
    )
>>>>>>> main

    # 2. 바로 회원가입 처리 (동기/비동기)
    try:
        create_user(user.user_id, user.name, user.password)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"회원 가입 실패: {str(e)}")

    # 3. 성공 응답
    return {"message": "회원 가입 완료"}

@router.post("/login", response_model=UserLoginResponse)
async def login(user: UserLoginRequest, response: Response):
    db_user = get_user_by_id(user.user_id)
    if not db_user:
        raise HTTPException(status_code=401, detail="존재하지 않는 ID입니다")
    
    if not verify_password(user.password, db_user['password']):
        raise HTTPException(status_code=401, detail="비밀번호가 일치하지 않습니다") 
    
    # 토큰-> 사용자 고유의 access할 수 있는 token 발급
    access_token = create_access_token(data={"sub": user.user_id})
    
    # 1. 응답 헤더에 Authorization 추가
    response.headers["Authorization"] = f"Bearer {access_token}"
    
    # token type bearer -> 이 토큰을 가진 사람은 인증된 사용자로 간주 
    # bearer: 소지자(bearer)가 토큰의 권한을 가진다 => http-only 쿠키를 사용할 때는 bearer 필요 x
    # 로그인 시 서버에 보낼 때 붙여서 사용 
    # return {"access_token": token, "token_type": "bearer", "message": f"{db_user['name']}님, 환영합니다!"}
    return UserLoginResponse(
        status="success",
        code=200,
        message=f"{db_user['name']}님, 환영합니다!"
    )

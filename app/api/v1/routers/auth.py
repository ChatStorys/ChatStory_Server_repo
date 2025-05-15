from fastapi import APIRouter, HTTPException, BackgroundTasks, Response, Depends
# 회원 가입
from app.schemas.user_schema import UserCreate
from app.redis.queue import queue
from app.services.user_service import create_user
# 로그인
from app.schemas.user_schema import UserLogin
from app.services.user_service import get_user_by_id, verify_password
from app.core.jwt import create_access_token
from passlib.context import CryptContext

router = APIRouter()
# bcrypt로 비밀번호 해싱 
pwd_context = CryptContext(schemes=['bcrypt'], deprecated = 'auto')

@router.post("/register")
async def register(user: UserCreate):
    if get_user_by_id(user.user_id):  # 이미 존재하는 사용자
        raise HTTPException(status_code=400, detail="이미 존재하는 ID입니다.")
    
    job = queue.enqueue(
        create_user,
        user.user_id,
        user.name,
        user.password
    )
    return {"job_id": job.id, "message" : "회원 가입 완료"}



@router.post("/login")
async def login(user: UserLogin, response: Response):
    db_user = get_user_by_id(user.user_id)
    if not db_user:
        raise HTTPException(status_code=401, detail="존재하지 않는 ID입니다")
    
    if not verify_password(user.password, db_user['password']):
        raise HTTPException(status_code=401, detail = "비밀번호가 일치하지 않습니다") 
    
    # 토큰-> 사용자 고유의 access할 수 있는 token 발급
    token = create_access_token(data={"sub": user.user_id})
    
    # HTTP-Only 쿠키 설정
    response.set_cookie(
        key="access_token",
        value=f"{token}", # Bearer X, http-only cookie를 서용할 때는 단순히 토큰 문자열만 저장
        httponly=True,
        max_age=3600,  # 1시간 유효기간
        samesite="Strict",
        secure=True  # HTTPS에서만 동작하도록 설정 (테스트 시에는 False로 변경 가능)
    )
    
    # token type bearer -> 이 토큰을 가진 사람은 인증된 사용자로 간주 
    # bearer: 소지자(bearer)가 토큰의 권한을 가진다 
    # 로그인 시 서버에 보낼 때 붙여서 사용 
    # return {"access_token": token, "token_type": "bearer", "message": f"{db_user['name']}님, 환영합니다!"}
    return {"message": f"{db_user['name']}님, 환영합니다!"}


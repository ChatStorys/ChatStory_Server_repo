from pydantic import BaseModel, Field

class UserCreate(BaseModel): # 회원 가입 모델
    name: str = Field(..., min_length=2)
    user_id: str = Field(..., min_length=4, max_length=20)
    password: str = Field(..., min_length=8)
    
class UserLogin(BaseModel): # 로그인 모델
    user_id: str
    password: str
    
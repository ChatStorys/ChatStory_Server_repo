from pydantic import BaseModel, Field

class UserCreate(BaseModel): # 회원 가입 모델
    name: str = Field(..., min_length=2)
    user_id: str = Field(..., min_length=4, max_length=20)
    password: str = Field(..., min_length=8)

class UserCreateResponse(BaseModel):
    status: str
    code: int
    user_id: str
    message: str
    
    
class UserLoginRequest(BaseModel): # 로그인 모델
    user_id: str
    password: str

class UserLoginResponse(BaseModel): # 로그인 모델
    status: str
    code: int
    message: str
    access_token: str
    token_type: str
    
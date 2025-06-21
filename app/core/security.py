from fastapi.middleware.cors import CORSMiddleware
from fastapi import fastapi

app.fastapi()

app.add_middleware(
    CORSMiddleware,
    aallow_origins=[frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Authorization"],  # ← 여기에 토큰 헤더 이름을 추가!
)

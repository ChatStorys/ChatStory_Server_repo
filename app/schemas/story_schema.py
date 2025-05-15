from pydantic import BaseModel, Field
from typing import Literal

class StoryCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)
    genre: Literal["동화", "연애", "액션", "판타지", "무협", "스릴러", "추리"]
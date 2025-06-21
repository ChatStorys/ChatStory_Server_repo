import os
import httpx
from app.db.mongo import db
from app.AI.schemas import (
    ChatMessageRequest, ChatMessageResponse,
    ChapterEndAIRequest, ChapterEndAIResponse,
    MusicItem
)

HF_API_URL = "https://hglww4g5jugd2khs.us-east-1.aws.endpoints.huggingface.cloud"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}

def send_message_to_ai_server(data: ChatMessageRequest) -> ChatMessageResponse:
    # chapter_text = db.stories.find_one({"bookId": data.book_id})["lastChapterText"]

    payload = {
        "user_id":      data.user_id,
        "user_message": data.user_message,
        "book_id":      data.book_id
    }

    url = f"{HF_API_URL}/story/continue"
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        out = resp.json()

    # out == {"status": "...", "code": 200, "message":"...","prompt":"..."}
    return ChatMessageResponse(**out)


def send_chapter_end_to_ai(req: ChapterEndAIRequest) -> ChapterEndAIResponse:
    """
    챕터 종료 시 AI에 본문을 보내 요약과 음악 추천을 받아옴
    """
    url = f"{HF_API_URL}/story/chapter/summary_with_music"
    payload = {
        "user_id": req.user_id,
        "book_id": req.book_id
    }

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            HF_API_URL,
            headers=headers,
            json=payload
        )
        resp.raise_for_status()
        data = resp.json()

    return ChapterEndAIResponse(
        status = data["status"],
        code = data["code"],
        summary= data.get("summary", ""),
        recommended_music=[
            MusicItem(**item) for item in data.get("recommended_music", [])
        ]
    )

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

    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": data.book_id}
        )
        response.raise_for_status()
        output = response.json()

    return ChatMessageResponse(
        status=output["status"],
        code=output["code"],
        message=output["message"],
        prompt=output["prompt"]
    )

def send_chapter_end_to_ai(req: ChapterEndAIRequest) -> ChapterEndAIResponse:
    """
    챕터 종료 시 AI에 본문을 보내 요약과 음악 추천을 받아옴
    """
    payload = {
        "bookId": req.book_id
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
        status=data.get("status", "success"),
        code=data.get("code", 200),
        summary=data["summary"],
        recommended_music=[MusicItem(**item) for item in data.get("recommended_music", [])]
    )

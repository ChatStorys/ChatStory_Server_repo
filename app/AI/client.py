import os
import httpx
from app.AI.schemas import ChatMessageRequest, ChatMessageResponse
from app.schemas.story_schema import ChapterEndAIRequest, ChapterEndAIResponse

HF_API_URL = "https://api-inference.huggingface.co/models/Jinuuuu/KoELECTRA_fine_tunning_emotion"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}

async def send_message_to_ai_server(data: ChatMessageRequest) -> ChatMessageResponse:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": data.message}
        )
        if response.status_code != 200:
            raise Exception(f"Hugging Face API error: {response.status_code}, {response.text}")
        
        output = response.json()
        
        # Hugging Face 감정분석 모델은 [{"label": "LABEL", "score": float}] 형식으로 반환함
        emotion = output[0]["label"] if isinstance(output, list) else "Unknown"
        return ChatMessageResponse(message=emotion)

# app/ai/client.py
import httpx
from app.AI.schemas import ChatMessageRequest, ChatMessageResponse
from app.core.config import settings  # .env에서 AI_SERVER_URL 불러올 예정
from app.schemas.story_schema import ChapterEndAIRequest, ChapterEndAIResponse
# from  import handle_story_continue, handle_chapter_summary_with_music

# AI 서버 코드 불러오기
def send_message_to_ai_server(data: ChatMessageRequest) -> ChatMessageResponse:
    result_text: str = handle_story_continue(
        user_id=data.user_id,
        message=data.message,
        book_id=data.book_id
    )
    
    return ChatMessageResponse(response= result_text)


def send_chapter_end_to_ai(data: ChapterEndAIRequest) -> ChapterEndAIResponse:
    summary, music = handle_chapter_summary_with_music(
        summary= data.summary,
        recommended_music= [
            {
            title: str,
            artist: str
            }
        ]
    )
    return ChapterEndAIResponse(summary=summary, music_rcommendation=music)


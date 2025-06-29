from pymongo import MongoClient, ReturnDocument, DESCENDING, ASCENDING
from datetime import datetime, timedelta, timezone
from app.db.mongo import db
# from app.AI.router import continue_story, ChatMessageRequest  # AI 요청 스키마
# from app.AI.schemas import StorySaveRequest
from app.schemas.story_schema import (
    StoryCreateRequest, FinishStoryRequest,
    ArchiveItemResponse, StoryContentResponse
)
from app.AI.schemas import ChapterContent, MusicItem
from typing import List, Dict
# from app.services.story_service import save_user_message
import os

mongo_url = os.getenv("MONGO_URL")
client = MongoClient(mongo_url)
db = client[os.getenv("DB_NAME", "chatstory")]

def get_next_sequence(name: str) -> int:
    """
    counters 컬렉션에서 name에 해당하는 문서의 seq 필드를 1 증가시키고,
    증가된 값을 반환합니다. upsert=True로 최초 호출 시 문서도 생성합니다.
    """
    counter = db.counters.find_one_and_update(
        {"_id": name},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER
    )
    return counter["seq"]

def create_story(user_id: str, request: StoryCreateRequest) -> str:
    # user_id = token["sub"]
    # 서버와 클라이언트가 서로 신뢰할 수 있게 주고받는 인증된 JSON
    
    # 1) 시퀀스 번호 얻기
    seq = get_next_sequence("bookId")
    # 2) bookId 문자열 포맷 (book001, book002, …)
    book_id = f"book{seq:03d}"
    
    # 소설 생성
    insert_in_Book = db.Book.insert_one({
        "title": request.title,
        "genre": request.category,
        "userId": user_id,
        "bookId": book_id,
        "workingFlag": True # 현재 작성 중인 챕터를 나타내는 Flag
        # 챕터 기록 뿐만 아니라 소설의 새로 쓰기, 이어 쓰기 중 하나를 택함
    })
    
    insert_in_Chapter = db.Chapter.insert_one({
        "chapter_Num": 1,
        "userId": user_id,
        "bookId": book_id,
        "sumChapter": "",
        "workingFlag" : True,
        "musicTitle": "",
        "composer": ""
    })
    
    insert_in_ChatStorage = db.ChatStorage.insert_one({
        "userId": user_id,
        "chapter_Num": 1,        
        "bookId": book_id, 
        "content": [
            { "User": "", "LLM_Model": "" }
            ]
    })
    
    return book_id

def get_creating_story(user_id: str):
    """
    작성 중인(완료되지 않은) 소설이 있으면 반환
    """
    return db.Book.find_one({
        "userId": user_id,
        "workingFlag": True
    })

# def send_front_chat(user_id: str, book_id: str, prompt: str) -> Dict:
#     """
#     프론트에서 받은 입력을 AI continue_story 함수에 직접 전달
#     """
#     ai_req = ChatMessageRequest(
#         user_id=user_id,
#         user_message=prompt,
#         book_id=book_id
#     )
#     # continue_story 는 dict 응답을 바로 반환합니다.
#     result = continue_story(ai_req)
#     return result

# def end_chapter(book_id: str, req: ChapterEndAIRequest) -> ChapterEndAIResponse:
#     # ai에 요청
#     ai_resp = send_chapter_end_to_ai(book_id, req)
#     # DB에 저장할 필요 없음
#     return ai_resp
    
def finish_story(user_id: str, book_id: str) -> bool:
    """
    user_id와 FinishStoryRequest(book_id, created_at, workingFlag)을 받아
    해당 소설의 workingFlag를 False로, 완료 시각(created_at)을 저장합니다.
    Returns True if update succeeded.
    """
    # 만약 req.created_at이 문자열이라면 datetime으로 파싱
    kst = timezone(timedelta(hours=9))
    completed_at = datetime.now(kst)
    # 1) Book 컬렉션: 진행 중이던 챕터만 workingFlag=False
    result = db.Book.update_one(
        {
            "userId": user_id,
            "bookId": book_id,
        },
        {
            "$set": {
                "workingFlag": False,
                "completedAt": completed_at,
            }
        }
    )
    # 2) Chapter 컬렉션: 진행 중이던 챕터만 workingFlag=False
    chap_flag_res = db.Chapter.update_one(
        {"userId": user_id, "bookId": book_id, "workingFlag": True},
        {"$set": {"workingFlag": False}}
    )
    
    # 3) 마지막 챕터 찾기, Chapter 콜렉션에 마지막 챕터 제거
    last_chap = db.Chapter.find_one(
        {"userId": user_id, "bookId": book_id},
        sort=[("chapter_Num", DESCENDING)]
    )
    if not last_chap:
        return True  # 챕터가 없으면 더 할 일 없음

    db.Chapter.delete_one({"_id": last_chap["_id"]})

    # 4) ChatStorage에서도 마지막 챕터 제거
    db.ChatStorage.update_one(
        {"userId": user_id, "bookId": book_id},
        {"$pull": {"content": {"chapter_Num": last_chap["chapter_Num"]}}}
    )
    
    
    return result.modified_count == 1

# 아카이브
def list_archived_stories(user_id: str) -> List[ArchiveItemResponse]:
    """
    완료된(workingFlag=False) 소설 목록을 반환합니다.
    각 항목에 bookId, title, createdAt 필드를 포함합니다.
    """
    # DB에서 작업 완료된 소설 문서 조회
    cursor = db.Book.find({
        "userId": user_id,
        "workingFlag": False
    }).sort("createdAt", DESCENDING)
    # Pydantic 모델로 변환하여 반환
    
    return [
        ArchiveItemResponse(
            bookId=doc["bookId"],
            title=doc["title"],
            createdAt=doc.get("createdAt") or doc["_id"].generation_time
        )
        for doc in cursor
    ]

def get_story_content(user_id: str, book_id: str) :
    """
    특정 소설(book_id)에 대한 전체 내용을 반환합니다.
    chapters 필드에 저장된 리스트를 그대로 내려줍니다.
    """
    book = db.Book.find_one({"userId": user_id, "bookId": book_id})
    if not book: # 찾는 책 없으면 None 반환
        return None
    
    # 여러 챕터 가져오기
    chapters_cursor = db.Chapter.find(
        {"userId": user_id, "bookId": book_id, "workingFlag": False}
    ).sort("chapter_Num", ASCENDING)


    chapters: List[ChapterContent] = []
    for chap in chapters_cursor:
        num = chap["chapter_Num"]

        # 3) ChatStorage 컬렉션에서 같은 챕터의 대화 저장 가져오기
        # ChatStorage에서 같은 챕터 문서를 통째로 조회
        chat_doc = db.ChatStorage.find_one(
            {"userId": user_id, "bookId": book_id, "chapter_Num": num},
            {"_id": 0, "content": 1}
        )
        msgs = chat_doc.get("content", []) if chat_doc else []

        # 4) messages를 “사용자: …\nAI: …\n” 형태로 합치기
        content_lines = []
        for m in msgs:
            if "User" in m:
                content_lines.append(f"{m['User']} ")
            elif "LLM_Model" in m:
                content_lines.append(f"{m['LLM_Model']} ")
        full_text = " ".join(content_lines)

        # 5) 추천 음악 정보 추출 (music 필드가 {musicTitle, composer} 형태로 저장되어 있다고 가정)
        music_info = None
        if chap.get("musicTitle") or chap.get("composer"):
            music_info = MusicItem(
                title=chap.get("musicTitle", ""),
                artist=chap.get("composer", "")
            )
            
        chapters.append(
            ChapterContent(
                chapter_num=num,
                content=full_text,
                recommended_music=[music_info] if music_info else []
            )
        )

    created_at = book.get("completedAt") or book["_id"].generation_time

    # ChatStorage 안에 있는 모든 chapter의 content들 이어붙여 한 번에 보내기

    return StoryContentResponse(
        bookId=book_id,
        title=book.get("title"),
        chapters=chapters,
        createdAt=created_at
    )


def delete_story(user_id: str, book_id: str) -> bool:
    """
    특정 소설(book_id)에 대한 문서를 삭제합니다.
    성공적으로 삭제되면 True를 반환합니다.
    """
    # 1) Book 문서 삭제
    book_res = db.Book.delete_one({
        "userId": user_id,
        "bookId": book_id
    })

    # 2) Chapter 문서들 삭제
    chap_res = db.Chapter.delete_many({
        "userId": user_id,
        "bookId": book_id
    })

    # 3) ChatStorage 문서 삭제
    # 만약 ChatStorage가 한 문서당 하나의 책만 저장한다면 delete_many
    chat_res = db.ChatStorage.delete_many({
        "userId": user_id,
        "bookId": book_id
    })
    try: 
        if (book_res.deleted_count == 1):
            return True
    except Exception as e:
        return f"삭제하다가 문제가 발생했습니다{e}"
        
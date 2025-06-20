# ChatStory Server

FastAPI 기반의 대화형 소설 생성 서비스 백엔드

## 주요 기능

* **소설 생성**: 사용자 인증 후, 장르와 제목을 입력하여 새 소설을 시작합니다. (작성 중인 소설이 있으면 이어쓰기 가능)
* **AI 챕터 작성**: 사용자가 채팅하듯 메시지를 보내면 AI가 다음 문단을 생성합니다.
* **챕터 요약·음악 추천**: 챕터 종료 시 AI 요약 및 분위기에 맞는 음악 추천 기능(현재는 더미 API 연동).
* **소설 완료 처리**: `workingFlag`를 `False`로 변경하여 소설 작성 완료 처리.
* **아카이브**: 완료된 소설 목록 조회, 상세 콘텐츠 조회, 소설 삭제 기능.

## 기술 스택

* **Language**: Python 3.10+
* **Framework**: FastAPI
* **Database**: MongoDB
* **인증**: JWT (RS256), HTTP Bearer
* **AI 통합**: 내부 더미 함수, 추후 외부 AI 서버 연동
* **라이브러리**: Pydantic, PyMongo, httpx

## 설치 및 실행

```bash
# Clone
git clone https://github.com/ChatStorys/ChatStory_Server_repo.git
cd ChatStory_Server_repo

# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정 (.env)
# MONGO_URL: MongoDB 연결 문자열
# DB_NAME: 사용 DB 이름 (기본: chatstory)
# JWT_PRIVATE_KEY, JWT_PUBLIC_KEY: 인증용 키

# 개발 서버 실행
uvicorn app.main:app --reload
```

## API 문서

자동 생성된 Swagger UI를 통해 확인 가능합니다:

```
http://localhost:8000/docs
```

### 주요 엔드포인트

| Method | Path                      | 설명              |
| ------ | ------------------------- | --------------- |
| POST   | `/story/create`           | 새 소설 생성         |
| POST   | `/story/chat/send`        | AI 챕터 생성 메시지 전송 |
| POST   | `/story/chapter/end`      | 챕터 종료 요약·음악 추천  |
| POST   | `/story/finish`           | 소설 완료 처리 (쿼리)   |
| GET    | `/story/archive`          | 완료된 소설 목록 조회    |
| GET    | `/story/archive/{bookId}` | 소설 상세 콘텐츠 조회    |
| DELETE | `/story/archive/{bookId}` | 소설 삭제           |

## 스키마

README 내 전체 스키마 정보 생략. Swagger UI 참조.

## 기여하기

1. Fork 프로젝트
2. 새로운 브랜치 생성: `git checkout -b feature/your-feature`
3. 커밋: `git commit -m "Add your feature"`
4. 푸시: `git push origin feature/your-feature`
5. Pull Request 생성

## 라이선스

MIT License © ChatStorys

```
```

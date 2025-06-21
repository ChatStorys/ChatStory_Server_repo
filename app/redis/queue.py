import os
import redis
from rq import Queue

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
conn = redis.Redis.from_url(redis_url)

queue = Queue(connection=conn)
# 이 파일은 필요 없어 보이긴 해
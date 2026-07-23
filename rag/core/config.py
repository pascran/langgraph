"""중앙 설정 — 128곳에 하드코딩되던 URL/모델/경로/청킹 파라미터를 환경변수 오버라이드 가능하게 단일화."""
import os

HF_HOME = os.environ.get("HF_HOME", "/home/coreedge-dev/langgraph/.hf")
os.environ.setdefault("HF_HOME", HF_HOME)

# 인프라 엔드포인트
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
GEN_URL = os.environ.get("GEN_URL", "http://localhost:8001/v1")
JUDGE_URL = os.environ.get("JUDGE_URL", "http://localhost:8002/v1")

# 모델
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
GEN_MODEL = os.environ.get("GEN_MODEL", "bottlecapai/ThinkingCap-Qwen3.6-27B-FP8")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen3-32B-AWQ")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# 컬렉션 / 데이터
COLLECTION = os.environ.get("COLLECTION", "silson_v2_sem")
DATA = os.environ.get("RAG_DATA", "data")
BLOCKS = f"{DATA}/parsed/blocks_kie.jsonl"
GOLDEN = f"{DATA}/golden/silson_golden.jsonl"

# 청킹(코사인급락)
CHUNK_PCTL = 88       # 경계 임계 백분위
CHUNK_MIN_SPLIT = 200  # 이 이하 누적에선 분리 안 함
CHUNK_MAX = 700        # 이 이상이면 강제 분리
SENT_MIN = 2           # 문장 최소길이(면책 리스트 보존 위해 낮춤)
CHUNK_MERGE_MIN = 30   # 이 미만 조각은 앞 청크에 병합
TOP_K = 5
PREFETCH = 30

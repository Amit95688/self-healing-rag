import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

# ============================================================
# FEATURE TOGGLES
# ============================================================
USE_QUERY_DECOMPOSITION = False
USE_HYDE = False
USE_WEB_FALLBACK = True

# ============================================================
# RETRIEVAL / GENERATION LIMITS
# ============================================================
TOP_N_RELEVANT = 2
RERANK_MIN_SCORE = 0.0
MAX_RETRIES = 1
MAX_REWRITE_TRIES = 1

# ============================================================
# PERSISTED ARTIFACTS
# ============================================================
INDEX_PATH = "./faiss_index"
CHUNKS_PATH = "./chunks_cache.pkl"
LOG_DIR = "./logs"

# ============================================================
# LLM - single free, low-latency MoE model for everything
# ============================================================
llm = ChatOpenAI(
    model="qwen/qwen3-next-80b-a3b-instruct",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_QWEN_API_KEY"),
    temperature=0,
)

# ============================================================
# EMBEDDINGS - small, local, CPU-friendly, free
# ============================================================
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

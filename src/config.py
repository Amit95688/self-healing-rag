import os
from typing import Any

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

def _build_llm() -> ChatOpenAI:
    api_key = (
        os.getenv("NVIDIA_QWEN_API_KEY")
        or os.getenv("NVIDIA_LLAMA_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_ADMIN_KEY")
    )

    if not api_key:
        raise RuntimeError(
            "Missing LLM credentials. Set NVIDIA_QWEN_API_KEY or NVIDIA_LLAMA_API_KEY "
            "for the NVIDIA endpoint, or set OPENAI_API_KEY / OPENAI_ADMIN_KEY."
        )

    return ChatOpenAI(
        model=os.getenv("NVIDIA_MODEL", "qwen/qwen3-next-80b-a3-instruct"),
        base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        api_key=api_key,
        temperature=0,
    )


class _LazyLLM:
    def __init__(self) -> None:
        self._client: ChatOpenAI | None = None

    def _get_client(self) -> ChatOpenAI:
        if self._client is None:
            self._client = _build_llm()
        return self._client

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_client(), name)


llm = _LazyLLM()

# ============================================================
# EMBEDDINGS - small, local, CPU-friendly, free
# ============================================================
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

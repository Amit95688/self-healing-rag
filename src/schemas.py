from typing import List, Literal, TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class State(TypedDict, total=False):
    question: str

    is_blocked: bool
    block_reason: str
    need_retrieval: bool

    sub_queries: List[str]
    search_texts: List[str]
    retrieval_query: str
    rewrite_tries: int

    docs: List[Document]
    relevant_docs: List[Document]
    context: str
    answer: str

    issup: Literal["fully_supported", "partially_supported", "no_support"]
    isuse: Literal["useful", "not_useful"]
    evidence: List[str]
    use_reason: str
    retries: int

    used_web_fallback: bool

    output_safe: bool
    output_block_reason: str


class GuardAndRouteDecision(BaseModel):
    allowed: bool = Field(..., description="True if the question is safe to process.")
    reason: str = Field(..., description="Short reason for the allow/block decision.")
    category: Literal[
        "safe",
        "prompt_injection",
        "jailbreak_attempt",
        "pii_exposure",
        "unsafe_content",
        "nonsense",
    ]
    should_retrieve: bool = Field(
        ..., description="True if answering requires looking up the reference textbooks."
    )


class DecomposeDecision(BaseModel):
    sub_queries: List[str] = Field(
        ...,
        description=(
            "1-3 atomic, self-contained search queries. If the question is already simple/atomic, "
            "return a list with just that one query."
        ),
    )


class HydeDecision(BaseModel):
    hypothetical_passages: List[str] = Field(
        ...,
        description=(
            "One hypothetical passage per input query, written as if it were an excerpt from a "
            "textbook answering that query."
        ),
    )


class VerifyDecision(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    isuse: Literal["useful", "not_useful"]
    evidence: List[str] = Field(default_factory=list)
    reason: str = Field(..., description="Short 1-line reason covering both checks.")


class RewriteDecision(BaseModel):
    retrieval_query: str = Field(
        ..., description="Rewritten query optimized for vector retrieval against ML/AI reference textbooks."
    )


class OutputGuardDecision(BaseModel):
    safe: bool = Field(..., description="True if the answer is safe to show as-is.")
    reason: str = Field(..., description="Short reason for the decision.")
    category: Literal["safe", "toxic_or_offensive", "pii_leak", "system_prompt_leak", "off_topic"]

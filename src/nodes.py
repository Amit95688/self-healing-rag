from typing import List, Literal

from langchain_core.documents import Document

from .config import (
    MAX_RETRIES,
    MAX_REWRITE_TRIES,
    SKIP_OUTPUT_GUARDRAIL,
    SKIP_VERIFY,
    USE_WEB_FALLBACK,
    llm,
)
from .prompts import (
    decompose_prompt,
    direct_generation_prompt,
    guard_and_route_prompt,
    hyde_prompt,
    output_guard_prompt,
    rag_generation_prompt,
    revise_prompt,
    rewrite_for_retrieval_prompt,
    verify_prompt,
    web_fallback_prompt,
)
from .retrieval import rerank_documents, retrieve_documents
from .schemas import (
    DecomposeDecision,
    GuardAndRouteDecision,
    HydeDecision,
    OutputGuardDecision,
    RewriteDecision,
    State,
    VerifyDecision,
)


def guard_and_route(state: State):
    decision: GuardAndRouteDecision = llm.with_structured_output(GuardAndRouteDecision).invoke(
        guard_and_route_prompt.format_messages(question=state["question"])
    )
    return {
        "is_blocked": not decision.allowed,
        "block_reason": decision.reason,
        "need_retrieval": decision.should_retrieve,
    }


def route_after_guard(state: State) -> Literal["blocked", "generate_direct", "retrieve"]:
    if state["is_blocked"]:
        return "blocked"
    return "retrieve" if state["need_retrieval"] else "generate_direct"


def blocked_response(state: State):
    return {
        "answer": (
            "I can't help with that request. "
            f"({state.get('block_reason', 'Blocked by input guardrail')})"
        )
    }


def generate_direct(state: State):
    out = llm.invoke(direct_generation_prompt.format_messages(question=state["question"]))
    return {"answer": out.content}


def decompose_query(state: State):
    decision: DecomposeDecision = llm.with_structured_output(DecomposeDecision).invoke(
        decompose_prompt.format_messages(question=state["question"])
    )
    sub_queries = decision.sub_queries[:3] if decision.sub_queries else [state["question"]]
    return {"sub_queries": sub_queries}


def hyde_expand(state: State):
    queries = state.get("sub_queries") or [state.get("retrieval_query") or state["question"]]
    queries_block = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(queries))

    decision: HydeDecision = llm.with_structured_output(HydeDecision).invoke(
        hyde_prompt.format_messages(queries_block=queries_block)
    )

    passages = decision.hypothetical_passages
    if len(passages) != len(queries):
        passages = list(queries)

    return {"search_texts": passages, "sub_queries": list(queries)}


def retrieve(state: State):
    search_texts = state.get("search_texts") or []
    if not search_texts:
        search_texts = state.get("sub_queries") or []
    if not search_texts:
        search_texts = [state.get("retrieval_query") or state["question"]]

    docs = retrieve_documents(search_texts)
    return {"docs": docs}


def rerank_docs(state: State):
    docs = state.get("docs", [])
    if not docs:
        return {"relevant_docs": []}

    relevant_docs = rerank_documents(state["question"], docs)
    return {"relevant_docs": relevant_docs}


def route_after_relevance(
    state: State,
) -> Literal["generate_from_context", "rewrite_question", "web_search_fallback", "no_answer_found"]:
    if state.get("relevant_docs"):
        return "generate_from_context"
    if state.get("rewrite_tries", 0) < MAX_REWRITE_TRIES:
        return "rewrite_question"
    if USE_WEB_FALLBACK and not state.get("used_web_fallback"):
        return "web_search_fallback"
    return "no_answer_found"


def generate_from_context(state: State):
    context_parts = []
    for d in state.get("relevant_docs", []):
        page = d.metadata.get("page", "NA")
        context_parts.append(f"[page {page}]\n{d.page_content}")

    context = "\n\n---\n\n".join(context_parts).strip()
    if not context:
        return {"answer": "No answer found.", "context": ""}

    out = llm.invoke(
        rag_generation_prompt.format_messages(question=state["question"], context=context)
    )
    return {"answer": out.content, "context": context}


def no_answer_found(state: State):
    return {"answer": "No answer found.", "context": ""}


def web_search_fallback(state: State):
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            hits = list(ddgs.text(state["question"], max_results=4))
    except Exception as e:
        print(f"  web search unavailable ({e}) - giving up.")
        return {"answer": "No answer found.", "context": "", "used_web_fallback": True}

    if not hits:
        return {"answer": "No answer found.", "context": "", "used_web_fallback": True}

    context_parts = []
    for h in hits:
        url = h.get("href") or h.get("url") or "unknown"
        context_parts.append(f"[Source: {url}]\n{h.get('title', '')}\n{h.get('body', '')}")
    context = "\n\n---\n\n".join(context_parts).strip()

    out = llm.invoke(web_fallback_prompt.format_messages(question=state["question"], context=context))
    return {"answer": out.content, "context": context, "used_web_fallback": True}


def route_after_web_fallback(state: State) -> Literal["verify_answer", "output_guardrail"]:
    if state.get("answer") in ("", "No answer found."):
        return "output_guardrail"
    return "verify_answer"


def verify_answer(state: State):
    if SKIP_VERIFY:
        return {
            "issup": "fully_supported",
            "isuse": "useful",
            "evidence": "",
            "use_reason": "skipped in CI fast mode",
        }

    decision: VerifyDecision = llm.with_structured_output(VerifyDecision).invoke(
        verify_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    )
    return {
        "issup": decision.issup,
        "isuse": decision.isuse,
        "evidence": decision.evidence,
        "use_reason": decision.reason,
    }


def route_after_verify(
    state: State,
) -> Literal["output_guardrail", "revise_answer", "rewrite_question", "web_search_fallback", "no_answer_found"]:
    issup = state.get("issup")
    isuse = state.get("isuse")

    if issup == "fully_supported" and isuse == "useful":
        return "output_guardrail"

    if issup == "no_support" and state.get("retries", 0) < MAX_RETRIES:
        return "revise_answer"

    if isuse == "not_useful" and state.get("rewrite_tries", 0) < MAX_REWRITE_TRIES:
        return "rewrite_question"

    if issup in ("fully_supported", "partially_supported"):
        return "output_guardrail"

    if USE_WEB_FALLBACK and not state.get("used_web_fallback"):
        return "web_search_fallback"

    return "no_answer_found"


def revise_answer(state: State):
    out = llm.invoke(
        revise_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    )
    return {"answer": out.content, "retries": state.get("retries", 0) + 1}


def rewrite_question(state: State):
    decision: RewriteDecision = llm.with_structured_output(RewriteDecision).invoke(
        rewrite_for_retrieval_prompt.format_messages(
            question=state["question"],
            retrieval_query=state.get("retrieval_query", ""),
            answer=state.get("answer", ""),
        )
    )
    return {
        "retrieval_query": decision.retrieval_query,
        "rewrite_tries": state.get("rewrite_tries", 0) + 1,
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "search_texts": [],
        "sub_queries": [],
    }


def output_guardrail(state: State):
    answer = (state.get("answer") or "").strip()

    if answer in ("", "No answer found."):
        return {"output_safe": True, "output_block_reason": ""}

    if SKIP_OUTPUT_GUARDRAIL:
        return {"output_safe": True, "output_block_reason": ""}

    decision: OutputGuardDecision = llm.with_structured_output(OutputGuardDecision).invoke(
        output_guard_prompt.format_messages(answer=answer)
    )

    if not decision.safe:
        return {
            "answer": "I can't share that response. Please rephrase your question.",
            "output_safe": False,
            "output_block_reason": decision.reason,
        }

    return {"output_safe": True, "output_block_reason": ""}

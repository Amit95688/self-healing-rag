from langgraph.graph import END, START, StateGraph

from .config import USE_HYDE, USE_QUERY_DECOMPOSITION
from .nodes import (
    blocked_response,
    decompose_query,
    generate_direct,
    generate_from_context,
    guard_and_route,
    hyde_expand,
    no_answer_found,
    output_guardrail,
    rerank_docs,
    retrieve,
    revise_answer,
    rewrite_question,
    route_after_guard,
    route_after_relevance,
    route_after_verify,
    route_after_web_fallback,
    verify_answer,
    web_search_fallback,
)
from .schemas import State


def build_app():
    g = StateGraph(State)
    g.add_edge(START, "guard_and_route")

    g.add_node("guard_and_route", guard_and_route)
    g.add_node("blocked_response", blocked_response)
    g.add_node("generate_direct", generate_direct)
    g.add_node("decompose_query", decompose_query)
    g.add_node("hyde_expand", hyde_expand)
    g.add_node("retrieve", retrieve)
    g.add_node("rerank_docs", rerank_docs)
    g.add_node("generate_from_context", generate_from_context)
    g.add_node("no_answer_found", no_answer_found)
    g.add_node("web_search_fallback", web_search_fallback)
    g.add_node("verify_answer", verify_answer)
    g.add_node("revise_answer", revise_answer)
    g.add_node("rewrite_question", rewrite_question)
    g.add_node("output_guardrail", output_guardrail)

    if USE_QUERY_DECOMPOSITION:
        retrieval_entry = "decompose_query"
    elif USE_HYDE:
        retrieval_entry = "hyde_expand"
    else:
        retrieval_entry = "retrieve"

    g.add_conditional_edges(
        "guard_and_route",
        route_after_guard,
        {
            "blocked": "blocked_response",
            "generate_direct": "generate_direct",
            "retrieve": retrieval_entry,
        },
    )
    g.add_edge("blocked_response", END)

    g.add_edge("generate_direct", "output_guardrail")

    if USE_QUERY_DECOMPOSITION:
        g.add_edge("decompose_query", "hyde_expand" if USE_HYDE else "retrieve")
    if USE_HYDE:
        g.add_edge("hyde_expand", "retrieve")

    g.add_edge("retrieve", "rerank_docs")
    g.add_conditional_edges(
        "rerank_docs",
        route_after_relevance,
        {
            "generate_from_context": "generate_from_context",
            "rewrite_question": "rewrite_question",
            "web_search_fallback": "web_search_fallback",
            "no_answer_found": "no_answer_found",
        },
    )
    g.add_edge("no_answer_found", "output_guardrail")

    g.add_conditional_edges(
        "web_search_fallback",
        route_after_web_fallback,
        {
            "verify_answer": "verify_answer",
            "output_guardrail": "output_guardrail",
        },
    )

    g.add_edge("generate_from_context", "verify_answer")
    g.add_conditional_edges(
        "verify_answer",
        route_after_verify,
        {
            "output_guardrail": "output_guardrail",
            "revise_answer": "revise_answer",
            "rewrite_question": "rewrite_question",
            "web_search_fallback": "web_search_fallback",
            "no_answer_found": "no_answer_found",
        },
    )

    g.add_edge("revise_answer", "verify_answer")
    g.add_edge("rewrite_question", "retrieve")
    g.add_edge("output_guardrail", END)

    return g.compile()


app = build_app()

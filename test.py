"""SentinelRAG eval runner."""

import argparse
import importlib
import json
import re
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_questions(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def import_pipeline(module_name: str):
    """
    Imports `app` from the user's pipeline file. The file is expected to sit
    next to this script (or be importable on sys.path) and to expose a
    compiled LangGraph object named `app`, matching the structure from the
    SentinelRAG draft (workflow.compile() -> app).
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        print(
            f"Could not import '{module_name}'. Pass --pipeline-module with "
            f"the filename (no .py) of your SentinelRAG script.\nError: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not hasattr(module, "app"):
        print(
            f"Module '{module_name}' has no `app` attribute. This script "
            f"expects a compiled LangGraph object (workflow.compile()) "
            f"named `app` at module level.",
            file=sys.stderr,
        )
        sys.exit(1)

    return module.app


def normalize(text: str) -> set[str]:
    """Lowercase, strip punctuation, split to a word set for overlap scoring."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    stopwords = {
        "the", "a", "an", "is", "of", "to", "and", "in", "that", "this",
        "are", "for", "or", "as", "by", "with", "be", "it", "on",
    }
    return {w for w in text.split() if w and w not in stopwords}


def keyword_overlap_score(predicted: str, ground_truth: str) -> float:
    """
    Jaccard overlap between predicted answer and ground truth, after
    stripping stopwords. Not semantic similarity -- a cheap proxy that needs
    no extra model call. Treat scores in isolation with caution; this is a
    floor, not a substitute for human review or RAGAS answer_correctness.
    """
    pred_words = normalize(predicted)
    truth_words = normalize(ground_truth)
    if not pred_words or not truth_words:
        return 0.0
    intersection = pred_words & truth_words
    union = pred_words | truth_words
    return len(intersection) / len(union) if union else 0.0


def _is_refusal(answer: str) -> bool:
    answer = (answer or "").strip().lower()
    return answer.startswith("i can't help") or answer.startswith("i can't share") or answer.startswith("no answer found")


def select_ci_subset(questions: list[dict], factual_count: int = 8) -> list[dict]:
    """Representative subset: evenly sampled factual + all refusal tests."""
    refusal = [q for q in questions if q.get("expects_refusal")]
    factual = [q for q in questions if not q.get("expects_refusal")]
    step = max(1, len(factual) // factual_count)
    sampled = factual[::step][:factual_count]
    return sampled + refusal


def run_deterministic_eval(app, questions: list[dict]) -> list[dict]:
    results = []
    for item in questions:
        question = item["question"]
        expects_refusal = item.get("expects_refusal", False)

        start = time.time()
        try:
            outcome = app.invoke({"question": question})
            error = None
        except Exception as e:
            outcome = {"answer": "", "context": "", "docs": [], "relevant_docs": []}
            error = str(e)
        latency = time.time() - start

        answer = outcome.get("answer", "")
        output_safe = outcome.get("output_safe", True)
        docs = outcome.get("relevant_docs") or outcome.get("docs") or []

        if expects_refusal:
            passed = _is_refusal(answer) or output_safe is False
            overlap = None  # comparing a refusal to "Answer not found" text is meaningless
        else:
            passed = bool(answer.strip()) and not _is_refusal(answer)
            overlap = keyword_overlap_score(answer, item.get("ground_truth", "")) if item.get("ground_truth") else None

        results.append({
            "id": item["id"],
            "question": question,
            "expects_refusal": expects_refusal,
            "passed": passed,
            "answer": answer,
            "ground_truth": item.get("ground_truth", ""),
            "keyword_overlap": overlap,
            "latency_sec": round(latency, 2),
            "error": error,
            "docs_retrieved": len(docs),
            "output_safe": output_safe,
        })

    return results


def run_ragas_eval(app, questions: list[dict], deterministic_results: list[dict]) -> dict | None:
    """
    Adds RAGAS faithfulness + context_precision on top of the deterministic
    pass. Skips rows that expect a refusal -- faithfulness/context_precision
    assume there's a grounded answer to score against retrieved context,
    which doesn't apply to "I don't know" responses.
    """
    try:
        ragas = importlib.import_module("ragas")
        ragas_metrics = importlib.import_module("ragas.metrics")
        datasets = importlib.import_module("datasets")
    except ImportError:
        print(
            "RAGAS scoring requires the `ragas` and `datasets` packages. "
            "Install with: pip install ragas datasets\n"
            "Skipping RAGAS scoring; deterministic results above are still valid.",
            file=sys.stderr,
        )
        return None

    rows = []
    for item, det in zip(questions, deterministic_results):
        if item.get("expects_refusal", False):
            continue
        if not det["passed"]:
            continue

        outcome = app.invoke({"question": item["question"], "retry_count": 0})
        contexts = [d.page_content for d in (outcome.get("relevant_docs") or outcome.get("docs") or [])]

        rows.append({
            "question": item["question"],
            "answer": outcome.get("answer", ""),
            "contexts": contexts if contexts else [""],
            "ground_truth": item["ground_truth"],
        })

    if not rows:
        print(
            "No rows eligible for RAGAS scoring (all questions either "
            "expected refusal or did not return 'success').",
            file=sys.stderr,
        )
        return None

    dataset = datasets.Dataset.from_list(rows)
    ragas_result = ragas.evaluate(
        dataset,
        metrics=[ragas_metrics.faithfulness, ragas_metrics.context_precision],
    )
    return ragas_result


def print_report(results: list[dict], ragas_result=None) -> float:
    """Returns the pass rate (0.0-1.0) so callers (e.g. main()) can gate on it."""
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    errors = sum(1 for r in results if r["error"])
    avg_latency = sum(r["latency_sec"] for r in results) / total if total else 0
    pass_rate = passed / total if total else 0.0

    print("\n" + "=" * 60)
    print("SentinelRAG Eval Report")
    print("=" * 60)

    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        refusal_tag = " [refusal-test]" if r["expects_refusal"] else ""
        print(f"\n[{mark}] {r['id']}{refusal_tag} -- latency={r['latency_sec']}s docs={r['docs_retrieved']}")
        print(f"  Q: {r['question']}")
        if r["error"]:
            print(f"  ERROR: {r['error']}")
        else:
            print(f"  A: {r['answer'][:150]}{'...' if len(r['answer']) > 150 else ''}")
            if r["keyword_overlap"] is not None:
                print(f"  keyword_overlap: {r['keyword_overlap']:.2f}")
            print(f"  output_safe: {r['output_safe']}")

    print("\n" + "-" * 60)
    print(f"Passed: {passed}/{total}  |  Errors: {errors}  |  Avg latency: {avg_latency:.2f}s")
    print(f"PASS_RATE={pass_rate:.4f}")  # fixed format -- grep-friendly for CI

    if ragas_result is not None:
        print("\n" + "-" * 60)
        print("RAGAS scores (faithfulness, context_precision):")
        print(ragas_result)

    print("=" * 60 + "\n")
    return pass_rate


def main():
    parser = argparse.ArgumentParser(description="Run eval against SentinelRAG pipeline")
    parser.add_argument(
        "--questions", default="eval_questions.json",
        help="Path to eval question set (default: eval_questions.json)"
    )
    parser.add_argument(
        "--pipeline-module", default="src.graph",
        help="Import path of your pipeline module exposing `app` (default: src.graph)"
    )
    parser.add_argument(
        "--ragas", action="store_true",
        help="Also run RAGAS faithfulness + context_precision scoring "
             "(extra LLM-judge call per eligible question, requires `pip "
             "install ragas datasets`)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to write full JSON results"
    )
    parser.add_argument(
        "--min-pass-rate", type=float, default=None,
        help="If set (0.0-1.0), exit with code 1 when pass rate falls below "
             "this threshold. E.g. --min-pass-rate 0.8 for an 80%% gate. "
             "If omitted, the script always exits 0 regardless of results "
             "(report-only mode)."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Run only the first N questions from the question file."
    )
    parser.add_argument(
        "--ci-subset", action="store_true",
        help="Run a balanced CI subset (~8 factual + all refusal tests) "
             "instead of the full question set."
    )
    args = parser.parse_args()

    if not Path(args.questions).exists():
        print(f"Question file not found: {args.questions}", file=sys.stderr)
        sys.exit(1)

    questions = load_questions(args.questions)
    if args.ci_subset:
        questions = select_ci_subset(questions)
        print(f"CI subset selected: {len(questions)} questions")
    elif args.limit is not None:
        questions = questions[: args.limit]
        print(f"Limited to first {len(questions)} questions")
    app = import_pipeline(args.pipeline_module)

    print(f"Running {len(questions)} questions against {args.pipeline_module}.app...")
    results = run_deterministic_eval(app, questions)

    ragas_result = None
    if args.ragas:
        print("\nRunning RAGAS scoring (additional LLM calls)...")
        ragas_result = run_ragas_eval(app, questions, results)

    pass_rate = print_report(results, ragas_result)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Full results written to {args.output}")

    if args.min_pass_rate is not None:
        if pass_rate < args.min_pass_rate:
            print(
                f"FAIL: pass rate {pass_rate:.2%} is below required "
                f"threshold {args.min_pass_rate:.2%}",
                file=sys.stderr,
            )
            sys.exit(1)
        else:
            print(
                f"OK: pass rate {pass_rate:.2%} meets required threshold "
                f"{args.min_pass_rate:.2%}"
            )
    # If --min-pass-rate was not passed, exit 0 regardless -- report-only mode.


if __name__ == "__main__":
    main()
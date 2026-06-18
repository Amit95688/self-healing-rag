import sys

from .graph import app


def ask(question: str):
    return app.invoke({"question": question})


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m src.main \"your question\"")
        return

    question = " ".join(sys.argv[1:]).strip()
    result = ask(question)
    print(result.get("answer", ""))


if __name__ == "__main__":
    main()

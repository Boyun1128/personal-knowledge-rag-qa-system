"""
rag_query.py — RAG Q&A CLI for the Personal Knowledge RAG system

Usage:
    # Single query mode
    python rag_query.py --query "What is the latest Gemma model?"
    python rag_query.py --query "..." --top-k 5 --model gemini/gemini-2.5-flash

    # Interactive multi-turn mode
    python rag_query.py

    # Show help
    python rag_query.py --help
"""

import argparse
import os
import sys

from dotenv import load_dotenv

# Load .env if present
load_dotenv()


def get_store():
    """Connect to VectorStore using env vars."""
    from rag_core import get_vector_store
    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
    return get_vector_store(persist_dir)


def run_single_query(query: str, top_k: int, model: str, store) -> None:
    from rag_core import retrieve, generate_answer, format_citations
    embedding_model = os.environ.get("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
    chunks = retrieve(query, store, top_k=top_k, embedding_model=embedding_model)

    if not chunks:
        print(
            "\n⚠️  No relevant information found in the knowledge base.\n"
            "   Make sure you have run: python data_update.py --rebuild\n"
        )
        return

    # Generate answer
    try:
        answer = generate_answer(query, chunks, history=[], model=model)
    except Exception as e:
        print(f"\n[ERROR] LLM call failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Print answer and citations
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    print(f"\n{answer}")
    print(format_citations(chunks))
    print()


def run_interactive(top_k: int, model: str, store) -> None:
    from rag_core import retrieve, generate_answer, format_citations
    embedding_model = os.environ.get("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

    print("\n" + "="*60)
    print("🤖 AI Technology Knowledge Base — Interactive Mode")
    print("="*60)
    print(f"   Model: {model} | Top-k: {top_k}")
    print("   Type your question, or 'exit'/'quit' to stop.\n")

    history: list[dict] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Goodbye]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("[Goodbye]")
            break

        # Retrieve relevant chunks
        chunks = retrieve(
            user_input, store, top_k=top_k, embedding_model=embedding_model
        )

        if not chunks:
            print(
                "\n⚠️  No relevant information found for this query.\n"
                "   Try rephrasing, or run: python data_update.py --rebuild\n"
            )
            # Still add to history so context is preserved
            history.append({"role": "user", "content": user_input})
            history.append({
                "role": "assistant",
                "content": "No relevant information found in the knowledge base for this query.",
            })
            continue

        # Generate answer — preserve history on LLM failure
        try:
            answer = generate_answer(user_input, chunks, history=history, model=model)
        except Exception as e:
            print(f"\n[ERROR] LLM call failed: {e}", file=sys.stderr)
            print("   Your conversation history is preserved. Please try again.\n")
            # Do NOT modify history on failure (Property 9)
            continue

        # Print answer and citations
        print(f"\nAssistant: {answer}")
        print(format_citations(chunks))
        print()

        # Append to history (only on success)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "rag_query.py — Query the AI Technology Knowledge Base.\n\n"
            "Single query mode: python rag_query.py --query 'Your question'\n"
            "Interactive mode:  python rag_query.py\n\n"
            "Requires: LITELLM_API_KEY and LITELLM_BASE_URL in .env"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        metavar="TEXT",
        help="Query string for single-query mode. If omitted, enters interactive mode.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="N",
        help="Number of chunks to retrieve (default: 5).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini/gemini-2.5-flash",
        metavar="MODEL",
        help="LiteLLM model string (default: gemini/gemini-2.5-flash).",
    )

    args = parser.parse_args()

    # Check required env vars
    api_key = os.environ.get("LITELLM_API_KEY", "").strip()
    if not api_key:
        print(
            "[ERROR] Missing required environment variable 'LITELLM_API_KEY'. "
            "Copy .env.example to .env and set your key.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Connect to VectorStore
    try:
        store = get_store()
    except Exception as e:
        print(f"[ERROR] Failed to connect to VectorStore: {e}", file=sys.stderr)
        sys.exit(1)

    if args.query:
        run_single_query(args.query, top_k=args.top_k, model=args.model, store=store)
    else:
        run_interactive(top_k=args.top_k, model=args.model, store=store)


if __name__ == "__main__":
    main()

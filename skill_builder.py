"""
skill_builder.py — Auto-generate skill.md by querying the RAG system

Usage:
    python skill_builder.py [--output skill.md] [--model gemini/gemini-2.5-flash]
    python skill_builder.py --help
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# ---------------------------------------------------------------------------
# Predefined section queries
# ---------------------------------------------------------------------------

SECTIONS = [
    {
        "heading": "Overview",
        "query": "What is the overall phenomenon of AI technology rapid evolution and why does it matter?",
    },
    {
        "heading": "Core Concepts",
        "query": "What are the core concepts in AI rapid evolution, knowledge anxiety, and adaptation pressure?",
    },
    {
        "heading": "Key Trends",
        "query": "What are the key trends in AI model release cadence and their impact on users?",
    },
    {
        "heading": "Key Entities",
        "query": "Who are the key organizations, models, and researchers in AI rapid evolution?",
    },
    {
        "heading": "Methodology & Best Practices",
        "query": "What methodologies and best practices help people cope with rapid AI change?",
    },
    {
        "heading": "Knowledge Gaps & Limitations",
        "query": "What are the limitations and gaps in current knowledge about AI rapid evolution?",
    },
    {
        "heading": "Example Q&A",
        "query": "Give examples of common questions and answers about keeping up with AI developments.",
    },
]

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "knowledge_base"

PLACEHOLDER = (
    "_No relevant information found in the knowledge base for this section. "
    "Run `python data_update.py --rebuild` to populate the index._"
)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def build_skill_md(model: str) -> str:
    """Query RAG for each section and assemble the skill.md content."""
    from rag_core import get_vector_store, retrieve, generate_answer, format_citations

    store = get_vector_store(PERSIST_DIR)
    source_doc_count = store.count()

    section_results: list[dict] = []
    all_chunks: list[dict] = []

    for section in SECTIONS:
        heading = section["heading"]
        query = section["query"]
        print(f"[skill_builder] Querying section: {heading} ...", file=sys.stderr)

        chunks = retrieve(
            query,
            store,
            top_k=10,
            threshold=0.0,
            embedding_model=EMBEDDING_MODEL,
        )

        if chunks:
            answer = generate_answer(query, chunks, history=[], model=model)
            all_chunks.extend(chunks)
        else:
            answer = PLACEHOLDER

        section_results.append({
            "heading": heading,
            "query": query,
            "answer": answer,
        })

    # Build Source References from all collected chunks (deduplicated by source+chunk_index)
    seen: set[tuple] = set()
    unique_chunks: list[dict] = []
    for c in all_chunks:
        key = (c.get("source", ""), c.get("chunk_index", 0))
        if key not in seen:
            seen.add(key)
            unique_chunks.append(c)

    # Sort by source then chunk_index for a clean listing
    unique_chunks.sort(key=lambda c: (c.get("source", ""), c.get("chunk_index", 0)))

    # ---------------------------------------------------------------------------
    # Assemble markdown
    # ---------------------------------------------------------------------------
    timestamp = datetime.now(timezone.utc).isoformat()

    lines: list[str] = []

    lines.append("# Skill: AI Technology Rapid Evolution — Knowledge Anxiety & Adaptation")
    lines.append("")
    lines.append("## Metadata")
    lines.append(f"- Generated: {timestamp}")
    lines.append(f"- RAG Model: {model}")
    lines.append(f"- Embedding Model: {EMBEDDING_MODEL}")
    lines.append(f"- Source Documents: {source_doc_count}")
    lines.append("")

    for sr in section_results:
        lines.append(f"## {sr['heading']}")
        lines.append(f'<!-- RAG Query: "{sr["query"]}" -->')
        lines.append("")
        lines.append(sr["answer"])
        lines.append("")

    # Source References section
    lines.append("## Source References")
    lines.append('<!-- RAG Query: aggregated from all retrieved chunk citations -->')
    lines.append("")
    if unique_chunks:
        for i, c in enumerate(unique_chunks, 1):
            similarity = c.get("similarity", 1.0 - c.get("distance", 0.0))
            lines.append(
                f"- [{i}] **{c.get('source', 'unknown')}** "
                f"(chunk {c.get('chunk_index', 0)}, "
                f"similarity: {similarity:.3f})"
            )
    else:
        lines.append(
            "_No source documents were retrieved. "
            "Run `python data_update.py --rebuild` to populate the index._"
        )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "skill_builder.py — Auto-generate skill.md from the RAG knowledge base.\n\n"
            "Queries the vector store with predefined section questions and synthesizes\n"
            "the answers into a structured skill document.\n\n"
            "Example:\n"
            "    python skill_builder.py\n"
            "    python skill_builder.py --output my_skill.md --model gemini/gemini-2.5-flash"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="skill.md",
        metavar="FILE",
        help="Output file path (default: skill.md).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini/gemini-2.5-flash",
        metavar="MODEL",
        help="LiteLLM model string (default: gemini/gemini-2.5-flash).",
    )

    args = parser.parse_args()

    # Check required env vars at startup (Req 8.4)
    # Support both GEMINI_API_KEY and LITELLM_API_KEY (Ollama/LiteLLM setup)
    api_key = os.environ.get("LITELLM_API_KEY", "").strip()
    if not api_key:
        print(
            "[ERROR] Missing required environment variable 'LITELLM_API_KEY'. "
            "Copy .env.example to .env and set your key.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[skill_builder] Building skill.md → {args.output}", file=sys.stderr)
    print(f"[skill_builder] Model: {args.model}", file=sys.stderr)

    try:
        content = build_skill_md(model=args.model)
    except Exception as e:
        print(f"[ERROR] Failed to build skill.md: {e}", file=sys.stderr)
        sys.exit(1)

    # Overwrite output file on each run (Req 7.5)
    output_path = Path(args.output)
    output_path.write_text(content, encoding="utf-8")
    print(f"[skill_builder] Written: {output_path} ({len(content)} chars)", file=sys.stderr)


if __name__ == "__main__":
    main()

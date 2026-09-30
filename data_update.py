"""
data_update.py — Data pipeline CLI for the Personal Knowledge RAG system

Usage:
    python data_update.py [--rebuild] [--chunk-size 512] [--overlap 50]
                          [--data-dir ./data] [--processed-dir ./data/processed]

Workflow:
    1. Load manifest.json (tracks MD5 hashes of processed files)
    2. For each .txt in data/processed/:
       - Compute MD5 hash
       - Skip if hash unchanged (incremental mode)
       - Chunk the text with overlap
       - Embed chunks via sentence-transformers
       - Upsert to FAISS vector store with IDs: {stem}_{chunk_index}
    3. Save updated manifest.json

With --rebuild:
    - Delete the FAISS index and metadata
    - Reset manifest.json
    - Re-index all documents from scratch
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present (silently ignore if missing)
load_dotenv()


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: Path) -> dict:
    """Load manifest.json; return empty dict if missing or corrupted."""
    if not manifest_path.exists():
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[WARNING] manifest.json corrupted or unreadable ({e}). Treating all files as new.", file=sys.stderr)
        return {}


def save_manifest(manifest: dict, manifest_path: Path) -> None:
    """Save manifest.json atomically."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def compute_md5(file_path: Path) -> str:
    """Compute MD5 hash of a file's contents."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - Normalize line endings
    - Collapse excessive blank lines (>2 consecutive)
    - Strip leading/trailing whitespace
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_raw_file(file_path: Path) -> str:
    """
    Load a raw file (.txt, .md, .pdf) and return cleaned plain text.
    PDF support requires pypdf2.
    """
    suffix = file_path.suffix.lower()

    if suffix in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return clean_text(f.read())

    elif suffix == ".pdf":
        try:
            import PyPDF2
            text_parts = []
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    text_parts.append(page_text)
            return clean_text("\n\n".join(text_parts))
        except ImportError:
            print(f"[WARNING] pypdf2 not installed. Skipping PDF: {file_path}", file=sys.stderr)
            return ""
        except Exception as e:
            print(f"[WARNING] Failed to parse PDF {file_path}: {e}", file=sys.stderr)
            return ""

    else:
        print(f"[WARNING] Unsupported file type: {file_path.suffix}. Skipping.", file=sys.stderr)
        return ""


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[dict]:
    """
    Split text into overlapping chunks by character count.

    Args:
        text: Input text to chunk.
        chunk_size: Maximum characters per chunk.
        overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        List of dicts: [{"text", "char_start", "char_end"}]
        Returns a single chunk if text is shorter than chunk_size.
    """
    if not text:
        return []

    if len(text) <= chunk_size:
        return [{"text": text, "char_start": 0, "char_end": len(text)}]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text_str = text[start:end]
        chunks.append({
            "text": chunk_text_str,
            "char_start": start,
            "char_end": end,
        })
        if end == len(text):
            break
        start = end - overlap  # overlap with previous chunk
        if start <= 0:
            start = end  # safety: avoid infinite loop

    return chunks


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full data update pipeline."""
    from rag_core import get_vector_store, embed_texts

    data_dir = Path(args.data_dir)
    processed_dir = Path(args.processed_dir)
    manifest_path = Path(args.manifest)
    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

    # --- Step 1: Process raw files (raw/ → processed/) ---
    raw_dir = data_dir / "raw"
    if raw_dir.exists():
        raw_files = [
            f for f in raw_dir.iterdir()
            if f.suffix.lower() in (".txt", ".md", ".pdf") and f.name != ".gitkeep"
        ]
        if raw_files:
            print(f"[data_update] Processing {len(raw_files)} raw file(s)...")
            processed_dir.mkdir(parents=True, exist_ok=True)
            for raw_file in raw_files:
                out_path = processed_dir / (raw_file.stem + ".txt")
                if out_path.exists() and not args.rebuild:
                    # Check if raw file changed
                    raw_md5 = compute_md5(raw_file)
                    manifest = load_manifest(manifest_path)
                    if manifest.get(f"raw_{raw_file.name}", {}).get("md5") == raw_md5:
                        continue  # unchanged
                text = load_raw_file(raw_file)
                if text:
                    out_path.write_text(text, encoding="utf-8")
                    print(f"  [clean] {raw_file.name} → {out_path.name}")

    # --- Step 2: Connect to VectorStore ---
    print(f"[data_update] Connecting to VectorStore at: {persist_dir}")
    try:
        store = get_vector_store(persist_dir)
    except Exception as e:
        print(f"[ERROR] Failed to connect to VectorStore: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Step 3: Handle --rebuild ---
    manifest = load_manifest(manifest_path)
    if args.rebuild:
        print("[data_update] --rebuild: clearing VectorStore and manifest...")
        try:
            store.clear()
        except Exception as e:
            print(f"[ERROR] Failed to clear VectorStore: {e}", file=sys.stderr)
            sys.exit(1)
        manifest = {}
        save_manifest(manifest, manifest_path)
        print("[data_update] Store cleared. Re-indexing all documents...")

    # --- Step 4: Scan processed/ and index new/changed files ---
    processed_dir.mkdir(parents=True, exist_ok=True)
    txt_files = sorted(processed_dir.glob("*.txt"))

    if not txt_files:
        print(f"[WARNING] No .txt files found in {processed_dir}. Nothing to index.", file=sys.stderr)
        return

    print(f"[data_update] Found {len(txt_files)} processed file(s). Checking for changes...")

    indexed_count = 0
    skipped_count = 0

    for txt_file in txt_files:
        file_key = txt_file.name
        current_md5 = compute_md5(txt_file)

        # Skip if unchanged (incremental mode)
        if not args.rebuild and manifest.get(file_key, {}).get("md5") == current_md5:
            skipped_count += 1
            continue

        # Load and chunk
        text = txt_file.read_text(encoding="utf-8", errors="replace")
        text = clean_text(text)
        if not text:
            print(f"  [WARNING] Empty file: {txt_file.name}. Skipping.", file=sys.stderr)
            continue

        chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
        if not chunks:
            continue

        # Embed all chunks in one batch
        chunk_texts = [c["text"] for c in chunks]
        try:
            embeddings = embed_texts(chunk_texts, model_name=embedding_model)
        except Exception as e:
            print(f"  [ERROR] Embedding failed for {txt_file.name}: {e}", file=sys.stderr)
            continue

        # Build IDs and metadata
        stem = txt_file.stem
        ids = [f"{stem}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": txt_file.name,
                "chunk_index": i,
                "char_start": chunks[i]["char_start"],
                "char_end": chunks[i]["char_end"],
            }
            for i in range(len(chunks))
        ]

        # Upsert to VectorStore (idempotent: same ID = skip if exists)
        try:
            store.upsert(
                ids=ids,
                documents=chunk_texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as e:
            print(f"  [ERROR] VectorStore upsert failed for {txt_file.name}: {e}", file=sys.stderr)
            continue

        # Update manifest
        manifest[file_key] = {
            "md5": current_md5,
            "chunk_count": len(chunks),
            "indexed": True,
        }
        save_manifest(manifest, manifest_path)

        print(f"  [indexed] {txt_file.name} → {len(chunks)} chunk(s)")
        indexed_count += 1

    print(
        f"\n[data_update] Done. "
        f"Indexed: {indexed_count} file(s), "
        f"Skipped (unchanged): {skipped_count} file(s). "
        f"Total chunks in store: {store.count()}"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "data_update.py — Build and update the RAG vector index.\n\n"
            "Reads processed .txt files from data/processed/, chunks them,\n"
            "embeds with sentence-transformers, and upserts to FAISS vector store.\n\n"
            "Default mode: incremental (only new/changed files).\n"
            "Use --rebuild to clear and re-index everything."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Clear the FAISS index and re-index all documents from scratch.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        metavar="N",
        help="Maximum characters per chunk (default: 512).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        metavar="N",
        help="Character overlap between consecutive chunks (default: 50).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        metavar="PATH",
        help="Root data directory (default: ./data).",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="./data/processed",
        metavar="PATH",
        help="Directory containing cleaned .txt files (default: ./data/processed).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="./manifest.json",
        metavar="PATH",
        help="Path to the manifest JSON file (default: ./manifest.json).",
    )

    args = parser.parse_args()

    # Validate overlap < chunk_size
    if args.overlap >= args.chunk_size:
        print(
            f"[ERROR] --overlap ({args.overlap}) must be less than --chunk-size ({args.chunk_size}).",
            file=sys.stderr,
        )
        sys.exit(1)

    run_pipeline(args)


if __name__ == "__main__":
    main()

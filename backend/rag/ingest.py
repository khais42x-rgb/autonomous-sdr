"""
Knowledge Base Ingestion — chunks Markdown files and embeds them into pgvector.
"""

import os
import glob
import logging
from dotenv import load_dotenv
from supabase import create_client
from backend.ai.embeddings import embed_batch

load_dotenv()
logger = logging.getLogger(__name__)

supabase_url = os.getenv("SUPABASE_URL", "")
supabase_key = os.getenv("SUPABASE_KEY", "")
supabase = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
KB_DIR = "knowledge"


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text.strip()] if text.strip() else []

    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk_words = words[i : i + size]
        chunk = " ".join(chunk_words)
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


def parse_filepath(filepath: str) -> tuple[str | None, str]:
    rel = filepath.replace(KB_DIR + "/", "").replace(KB_DIR + "\\", "")
    parts = rel.split(os.sep)
    if len(parts) < 2:
        return None, "general"
    campaign_slug = parts[0] if parts[0] != "global" else None
    doc_type = parts[1] if len(parts) > 1 else "general"
    return campaign_slug, doc_type


def ingest_directory(base_path: str = KB_DIR):
    if not supabase:
        print("Supabase credentials not configured yet. Skipping DB storage.")
        return

    files = glob.glob(f"{base_path}/**/*.md", recursive=True)
    print(f"Found {len(files)} Markdown files in {base_path}")

    all_chunks = []
    all_metadata = []

    for filepath in files:
        campaign_id, doc_type = parse_filepath(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({
                "campaign_id": campaign_id,
                "doc_type": doc_type,
                "source": filepath,
                "content": chunk,
            })

    if not all_chunks:
        print("No document chunks to ingest.")
        return

    print(f"Embedding {len(all_chunks)} chunks...")
    embeddings = embed_batch(all_chunks)

    rows = [
        {**meta, "embedding": emb}
        for meta, emb in zip(all_metadata, embeddings)
    ]

    try:
        supabase.table("kb_chunks").insert(rows).execute()
        print(f"Successfully ingested {len(all_chunks)} chunks into Supabase!")
    except Exception as e:
        print(f"Database insertion error: {e}")


if __name__ == "__main__":
    ingest_directory()
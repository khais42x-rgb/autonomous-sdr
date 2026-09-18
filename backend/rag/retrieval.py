"""
Knowledge Base Retrieval — semantic search over KB chunks.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv
from supabase import create_client
from backend.ai.embeddings import embed

load_dotenv()

supabase_url = os.getenv("SUPABASE_URL", "")
supabase_key = os.getenv("SUPABASE_KEY", "")
supabase = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None


@dataclass
class KBChunk:
    id: str
    content: str
    source: str
    doc_type: str
    campaign_id: str | None


def search_kb(
    query: str,
    campaign_id: str | None = None,
    doc_types: list[str] | None = None,
    limit: int = 5,
) -> list[KBChunk]:
    """
    Semantic vector search over kb_chunks table.
    """
    if not supabase:
        return []

    query_embedding = embed(query)

    try:
        result = supabase.rpc("match_kb_chunks", {
            "query_embedding": query_embedding,
            "match_campaign_id": campaign_id,
            "match_doc_types": doc_types,
            "match_limit": limit,
        }).execute()

        return [
            KBChunk(
                id=str(row["id"]),
                content=row["content"],
                source=row["source"],
                doc_type=row["doc_type"],
                campaign_id=row.get("campaign_id"),
            )
            for row in result.data
        ]
    except Exception as e:
        print(f"Vector search notice (using empty context until Supabase RPC is created): {e}")
        return []
"""
Grounding Check — fact-checks a draft message against its sources.
Runs BEFORE any message is sent to a prospect.
"""

from pydantic import BaseModel, Field
from backend.ai.llm import call_structured


class Claim(BaseModel):
    claim: str = Field(description="The specific factual claim from the message")
    supported: bool = Field(description="Whether this claim is supported by the sources")
    source_id: str | None = Field(
        default=None,
        description="Which source supports this claim, or null if unsupported"
    )


class GroundingOutput(BaseModel):
    claims: list[Claim] = Field(default_factory=list)
    all_supported: bool = Field(description="True if every claim is supported")
    risky_topics: list[str] = Field(
        default_factory=list,
        description="Any of: pricing, legal, commitment, guarantee, "
                    "specific_customer_name, performance_claim"
    )
    reason: str = Field(description="One-line summary of the grounding check")


SYSTEM_PROMPT = """You are a strict fact-checker for Sarvam AI's sales outreach.

Given a MESSAGE and its SOURCES, check every factual claim in the message.

A claim is "supported" only if the exact information appears in the sources.
General industry knowledge does NOT count as a source — only the provided text.

Flag these as risky_topics if they appear in the message:
- "pricing": any mention of specific prices, costs, or pricing tiers
- "legal": any legal claims, compliance certifications, or data residency guarantees
- "commitment": any promises about SLAs, uptime, or delivery timelines
- "guarantee": any performance guarantees or money-back offers
- "specific_customer_name": naming a customer not mentioned in the sources
- "performance_claim": specific accuracy numbers, latency figures, or benchmarks

Be strict. It is better to flag a claim as unsupported than to let a hallucination 
reach a prospect."""


def check(
    message: str,
    sources: list[dict],
    prompt_version: str | None = None,
) -> tuple[GroundingOutput, dict]:
    system = prompt_version or SYSTEM_PROMPT

    if sources:
        sources_text = "\n\n".join(
            f"[{s.get('id', i)}] {s.get('content', '')}"
            for i, s in enumerate(sources)
        )
    else:
        sources_text = "(No sources provided — all factual claims should be flagged as unsupported)"

    user = f"""MESSAGE:
{message}

SOURCES:
{sources_text}"""

    output, meta = call_structured(system, user, GroundingOutput, tier="small")
    meta["reason"] = output.reason
    meta["prompt_version"] = prompt_version or "default"
    return output, meta
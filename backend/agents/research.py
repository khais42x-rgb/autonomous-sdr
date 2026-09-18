"""
Research Agent — enriches a prospect with company and role context.
Triggered when a new prospect is discovered.
"""

from pydantic import BaseModel, Field
from backend.ai.llm import call_structured


class Fact(BaseModel):
    claim: str = Field(description="A specific factual claim about the company or prospect")
    source_url: str = Field(default="", description="URL where this fact was found")


class ResearchInput(BaseModel):
    prospect_name: str
    prospect_role: str
    company_name: str
    company_domain: str
    raw_notes: str = Field(default="", description="Any pre-existing notes or scraped data")


class ResearchOutput(BaseModel):
    company_summary: str = Field(description="2-3 sentence company overview")
    industry: str
    company_size: str = Field(description="e.g. '1000-5000', 'startup <50'")
    recent_news: list[str] = Field(default_factory=list, description="Recent developments")
    likely_pains: list[str] = Field(description="Likely pain points relevant to Sarvam AI")
    languages_served: list[str] = Field(
        default_factory=list,
        description="Languages the company's customers likely speak"
    )
    facts: list[Fact] = Field(default_factory=list)
    reason: str = Field(description="One-line summary of research findings")


SYSTEM_PROMPT = """You are a research agent for Sarvam AI's sales team.
Sarvam AI builds Indic-language AI products: speech-to-text (Saaras) for all 22 
scheduled Indian languages, text-to-speech (Bulbul), voice agents (Samvaad), 
and document AI (DocAgent). They focus on BFSI, gov-tech, and defence.

Given a prospect's name, role, and company, produce a structured research brief.
Focus on information that would help an SDR personalise outreach about Sarvam's products.

Rules:
- Only include facts you are confident about based on the provided data
- If raw_notes are provided, extract facts from them
- For likely_pains, think about what problems Sarvam's products could solve for this company
- For languages_served, consider the company's customer base geography
- Keep company_summary under 3 sentences
- Be specific, not generic"""


def run(input: ResearchInput, prompt_version: str | None = None) -> tuple[ResearchOutput, dict]:
    system = prompt_version or SYSTEM_PROMPT
    user = f"""Research this prospect:

Name: {input.prospect_name}
Role: {input.prospect_role}
Company: {input.company_name}
Domain: {input.company_domain}

Additional notes:
{input.raw_notes or "None provided."}"""

    output, meta = call_structured(system, user, ResearchOutput, tier="small")
    meta["reason"] = output.reason
    meta["prompt_version"] = prompt_version or "default"
    return output, meta
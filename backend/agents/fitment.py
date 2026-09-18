"""
ICP Fitment Agent — qualifies or rejects prospects against campaign ICP criteria.
Triggered after research is complete.
"""

from pydantic import BaseModel, Field
from backend.ai.llm import call_structured


class FitmentInput(BaseModel):
    campaign_id: str
    campaign_name: str
    prospect_name: str
    prospect_role: str
    company_name: str
    research_summary: str
    icp_criteria: str = Field(description="The campaign's ICP definition text")


class FitmentOutput(BaseModel):
    verdict: str = Field(
        description="'fit' (score >= 70), 'no_fit' (score < 40), or 'review' (40-69)"
    )
    score: int = Field(ge=0, le=100, description="Fit score 0-100")
    criteria_met: list[str] = Field(description="Which ICP criteria this prospect meets")
    criteria_missed: list[str] = Field(description="Which ICP criteria are not met or unclear")
    reason: str = Field(description="One-line plain English explanation")


SYSTEM_PROMPT = """You are an ICP qualification agent for Sarvam AI.

Given a prospect's research summary and the campaign's ICP criteria, evaluate 
whether this prospect is a good fit for outreach.

Scoring rules:
- 70-100: Strong fit. The prospect clearly matches the ICP and has likely pain points.
  Verdict: "fit"
- 40-69: Possible fit. Some criteria match but key ones are unclear or marginal.
  Verdict: "review" (a human should decide)
- 0-39: Poor fit. The prospect clearly doesn't match the ICP.
  Verdict: "no_fit"

Be specific about which criteria are met and which are missed.
If the data is insufficient to judge a criterion, list it in criteria_missed 
with a note like "insufficient data"."""


def run(input: FitmentInput, prompt_version: str | None = None) -> tuple[FitmentOutput, dict]:
    system = prompt_version or SYSTEM_PROMPT
    user = f"""Campaign: {input.campaign_name} ({input.campaign_id})

ICP Criteria:
{input.icp_criteria}

Prospect: {input.prospect_name}, {input.prospect_role} at {input.company_name}

Research Summary:
{input.research_summary}"""

    output, meta = call_structured(system, user, FitmentOutput, tier="small")

    # Enforce verdict-score consistency
    if output.score >= 70 and output.verdict != "fit":
        output.verdict = "fit"
    elif output.score < 40 and output.verdict != "no_fit":
        output.verdict = "no_fit"
    elif 40 <= output.score < 70 and output.verdict != "review":
        output.verdict = "review"

    meta["reason"] = output.reason
    meta["prompt_version"] = prompt_version or "default"
    return output, meta
"""
Follow-up Agent — decides when and how to follow up with unresponsive prospects.
"""

from pydantic import BaseModel, Field
from backend.ai.llm import call_structured


class FollowUpInput(BaseModel):
    campaign_name: str
    prospect_name: str
    prospect_role: str
    company_name: str
    touches_so_far: int
    days_since_last_touch: int
    last_channel: str
    last_result: str
    channel_policy: str


class FollowUpOutput(BaseModel):
    follow_up: bool
    channel: str = Field(default="")
    angle: str = Field(default="")
    wait_hours: int = Field(default=0)
    reason: str


SYSTEM_PROMPT = """You are a follow-up agent for Sarvam AI's SDR system.
Decide whether and how to follow up with a prospect who hasn't responded."""


def run(input: FollowUpInput, prompt_version: str | None = None) -> tuple[FollowUpOutput, dict]:
    system = prompt_version or SYSTEM_PROMPT
    user = f"Campaign: {input.campaign_name}\nProspect: {input.prospect_name}\nTouches: {input.touches_so_far}"
    output, meta = call_structured(system, user, FollowUpOutput, tier="small")
    meta["reason"] = output.reason
    return output, meta
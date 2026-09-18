"""
Voice Post-Processor — extracts structured data from call transcripts.
"""

from pydantic import BaseModel, Field
from backend.ai.llm import call_structured


class VoicePostInput(BaseModel):
    prospect_name: str
    company_name: str
    campaign_name: str
    call_goal: str
    transcript: str


class VoicePostOutput(BaseModel):
    qualified: bool
    outcome: str
    objections: list[str] = Field(default_factory=list)
    meeting_time: str | None = Field(default=None)
    key_interests: list[str] = Field(default_factory=list)
    next_step: str
    summary: str
    reason: str


SYSTEM_PROMPT = """You are a call analysis agent for Sarvam AI's SDR system.
Extract structured information from a sales call transcript."""


def run(input: VoicePostInput, prompt_version: str | None = None) -> tuple[VoicePostOutput, dict]:
    system = prompt_version or SYSTEM_PROMPT
    user = f"Campaign: {input.campaign_name}\nTranscript:\n{input.transcript}"
    output, meta = call_structured(system, user, VoicePostOutput, tier="small")
    meta["reason"] = output.reason
    return output, meta
"""
Conversation Agent — classifies inbound replies and suggests next steps.
"""

from pydantic import BaseModel, Field
from backend.ai.llm import call_structured


class ConversationInput(BaseModel):
    prospect_name: str
    campaign_name: str
    reply_text: str
    thread_history: str = Field(default="")
    channel: str = Field(default="email")


class ConversationOutput(BaseModel):
    intent: str = Field(
        description="interested, question, objection, not_now, unsubscribe, wrong_person, ooo, positive, negative, unclear"
    )
    sentiment: str = Field(description="'positive', 'neutral', or 'negative'")
    needs_human: bool = Field(description="True if a human should handle this")
    suggested_reply: str = Field(default="")
    referral_name: str | None = Field(default=None)
    objection_type: str | None = Field(default=None)
    reason: str = Field(description="One-line explanation")


SYSTEM_PROMPT = """You are a conversation agent for Sarvam AI's SDR system.
Classify inbound prospect replies and determine the appropriate next action."""


def run(input: ConversationInput, prompt_version: str | None = None) -> tuple[ConversationOutput, dict]:
    system = prompt_version or SYSTEM_PROMPT
    user = f"Campaign: {input.campaign_name}\nProspect: {input.prospect_name}\nReply: {input.reply_text}"
    output, meta = call_structured(system, user, ConversationOutput, tier="small")
    meta["reason"] = output.reason
    return output, meta
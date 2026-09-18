"""
Outreach Strategy Agent — the single decision-maker per prospect.
Looks at the full cross-channel timeline and picks the next best action.
"""

from pydantic import BaseModel, Field
from backend.ai.llm import call_structured


class TimelineEvent(BaseModel):
    day: int
    channel: str
    action: str
    result: str = ""


class StrategyInput(BaseModel):
    campaign_id: str
    campaign_name: str
    prospect_name: str
    prospect_role: str
    company_name: str
    prospect_summary: str = Field(description="Compact research + fitment summary")
    timeline: list[TimelineEvent] = Field(
        default_factory=list,
        description="Full cross-channel history, chronological"
    )
    channel_policy: str = Field(
        description="e.g. 'linkedin, email, call(after engagement only)'"
    )
    touches_this_week: int = 0
    max_touches_per_week: int = 3
    latest_reply: str = Field(default="", description="If prospect replied, the reply text")
    reply_intent: str = Field(default="", description="Classified intent of latest reply")


class StrategyOutput(BaseModel):
    action: str = Field(
        description="One of: send, call, wait, stop, escalate"
    )
    channel: str = Field(
        default="",
        description="Which channel to use (email, linkedin, whatsapp, call). Empty if wait/stop."
    )
    send_after_hours: int = Field(
        default=0,
        description="If action=wait, how many hours to wait before next action"
    )
    intent: str = Field(
        description="What the outreach should accomplish: introduce, follow_up, "
                    "handle_objection, book_meeting, re-engage, etc."
    )
    reason: str = Field(description="One-line plain English explanation of the decision")


SYSTEM_PROMPT = """You are the Outreach Strategy agent — the single SDR brain for each prospect.
You sell Sarvam AI products (voice agents, speech APIs, document AI for Indian languages).

You see the prospect's full cross-channel timeline and must decide the ONE next best action.

Available actions:
- "send": Send a message on the specified channel
- "call": Initiate a voice call (only if channel policy allows and prospect has engaged)
- "wait": Do nothing now, try again later (specify hours)
- "stop": Prospect is not interested or fully converted; stop outreach
- "escalate": Something needs human attention (pricing question, complex objection, etc.)

Decision rules:
1. Respect the channel policy strictly. Never use a channel that isn't allowed.
2. Never repeat a channel that was ignored twice in a row — switch channels or wait.
3. Calls only after the prospect has shown engagement (opened emails, replied, accepted connection).
4. If touches_this_week >= max_touches_per_week, action must be "wait".
5. If the prospect replied with "unsubscribe" or "stop", action is "stop".
6. If the prospect asked about pricing, action is "escalate" (never invent prices).
7. If the prospect replied positively, action is "call" or "send" to book a meeting.
8. First touch should be the lowest-friction channel in the policy (usually LinkedIn or email).
9. Space touches at least 24 hours apart unless the prospect is actively engaging.
10. After 5+ touches with no engagement, consider "stop".

Be concise in your reason. Think like an experienced SDR, not a bot."""


def run(input: StrategyInput, prompt_version: str | None = None) -> tuple[StrategyOutput, dict]:
    system = prompt_version or SYSTEM_PROMPT

    if input.timeline:
        timeline_str = "\n".join(
            f"[Day {e.day}] {e.channel}: {e.action}"
            + (f" — {e.result}" if e.result else "")
            for e in input.timeline
        )
    else:
        timeline_str = "(No previous touches — this is a new prospect)"

    user = f"""Campaign: {input.campaign_name}
Prospect: {input.prospect_name}, {input.prospect_role} at {input.company_name}
Summary: {input.prospect_summary}

Channel policy: {input.channel_policy}
Touches this week: {input.touches_this_week}/{input.max_touches_per_week}

Timeline:
{timeline_str}"""

    if input.latest_reply:
        user += f"\n\nLatest reply from prospect (intent: {input.reply_intent}):\n{input.latest_reply}"

    output, meta = call_structured(system, user, StrategyOutput, tier="small")
    meta["reason"] = output.reason
    meta["prompt_version"] = prompt_version or "default"
    return output, meta
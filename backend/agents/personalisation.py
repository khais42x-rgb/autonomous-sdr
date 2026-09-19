"""
Personalisation Agent — writes the actual outreach message.
Uses prospect context + KB chunks to generate grounded, personalised content.
Supports personalization tiers (template, ai, visual) and visual asset overlays.
"""

from pydantic import BaseModel, Field
from backend.ai.llm import call_structured


class VisualAsset(BaseModel):
    type: str = Field(default="image", description="'image' or 'video_thumbnail'")
    template_asset_id: str = Field(default="default_banner", description="ID of the base template to use")
    overlay_fields: dict[str, str] = Field(
        default_factory=dict,
        description="Dynamic text to overlay, e.g. {'prospect_name': 'Rajesh', 'company_name': 'HDFC Bank'}"
    )


class PersonalisationInput(BaseModel):
    campaign_name: str
    prospect_name: str
    prospect_role: str
    company_name: str
    research_facts: str = Field(description="Key facts from research agent")
    intent: str = Field(description="What this message should accomplish")
    channel: str = Field(description="email, linkedin, whatsapp")
    personalization_tier: str = Field(default="ai", description="template, ai, or visual")
    rep_name: str = Field(default="Sarvam SDR Team")
    rep_signature: str = Field(default="Sarvam AI · Indic AI Infrastructure")
    kb_context: str = Field(
        default="",
        description="Relevant knowledge base chunks (case studies, examples)"
    )
    previous_messages: str = Field(
        default="",
        description="Previous messages in this thread, if any"
    )
    feedback: str = Field(
        default="",
        description="Feedback from grounding check, if regenerating"
    )


class PersonalisationOutput(BaseModel):
    personalization_tier: str = Field(
        default="ai",
        description="One of: 'template', 'ai', or 'visual'"
    )
    subject: str = Field(default="", description="Email subject line (empty for non-email)")
    body: str = Field(description="The full message body")
    visual_asset: VisualAsset | None = Field(
        default=None,
        description="Optional image/video overlay specs if tier=='visual'"
    )
    facts_used: list[str] = Field(
        default_factory=list,
        description="List of specific facts/claims used in the message"
    )
    kb_chunks_used: list[str] = Field(
        default_factory=list,
        description="IDs or descriptions of KB chunks referenced"
    )
    reason: str = Field(description="One-line: why this message was written this way")


SYSTEM_PROMPT = """You are a personalisation agent writing outreach for Sarvam AI.
Sarvam builds Indic-language AI: speech-to-text (Saaras, 22 Indian languages), 
text-to-speech (Bulbul), voice agents (Samvaad), and document AI (DocAgent).

Write a message that:
1. Opens with something specific about the prospect (their company, role, recent news)
2. Connects their likely pain point to a specific Sarvam product
3. Includes a clear, low-friction call to action
4. Matches the tone of the channel (email = professional, WhatsApp = casual, LinkedIn = brief)

PERSONALIZATION TIERS:
- If personalization_tier is "template", just output the body with the exact merge tags provided.
- If personalization_tier is "ai", write a fully custom LLM-generated message.
- If personalization_tier is "visual", write the message AND populate the visual_asset object 
  with the prospect's name and company name for the image overlay.

CRITICAL RULES:
- ONLY use facts provided in the research_facts and kb_context. NEVER invent statistics, 
  customer names, pricing, or product features not in the provided context.
- Keep emails under 150 words. LinkedIn messages under 80 words. WhatsApp under 60 words.
- No generic openers like "I hope this email finds you well."
- Sign off with the rep's name and signature if provided."""


def run(input: PersonalisationInput, prompt_version: str | None = None) -> tuple[PersonalisationOutput, dict]:
    system = prompt_version or SYSTEM_PROMPT

    user = f"""Campaign: {input.campaign_name}
Channel: {input.channel}
Intent: {input.intent}
Personalization Tier: {input.personalization_tier}

Prospect: {input.prospect_name}, {input.prospect_role} at {input.company_name}

Research Facts:
{input.research_facts}

Knowledge Base Context:
{input.kb_context or "No specific KB chunks retrieved."}"""

    if input.previous_messages:
        user += f"\n\nPrevious messages in thread:\n{input.previous_messages}"

    if input.rep_name:
        user += f"\n\nSign off as: {input.rep_name}\n{input.rep_signature}"

    if input.feedback:
        user += f"\n\nIMPORTANT FEEDBACK FROM PREVIOUS DRAFT: {input.feedback}"

    output, meta = call_structured(
        system, user, PersonalisationOutput,
        tier="strong",
        temperature=0.7,
    )
    
    # Ensure the output tier matches the requested tier if the LLM gets confused
    if output.personalization_tier != input.personalization_tier:
        output.personalization_tier = input.personalization_tier

    meta["reason"] = output.reason
    meta["prompt_version"] = prompt_version or "default"
    return output, meta
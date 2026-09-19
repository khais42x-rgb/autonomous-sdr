from backend.agents.research import ResearchInput, run as run_research
from backend.agents.fitment import FitmentInput, run as run_fitment
from backend.agents.strategy import StrategyInput, run as run_strategy
from backend.agents.personalisation import PersonalisationInput, run as run_personalisation
from backend.agents.grounding import check as check_grounding

print("1. Researching Prospect...")
r_out, _ = run_research(ResearchInput(
    prospect_name="Rajesh Kumar", prospect_role="CIO", company_name="HDFC Bank", company_domain="hdfcbank.com"
))

print("2. Qualifying Prospect (Fitment)...")
f_out, _ = run_fitment(FitmentInput(
    campaign_id="cmp_a", campaign_name="BFSI Voice", prospect_name="Rajesh Kumar", prospect_role="CIO",
    company_name="HDFC Bank", research_summary=r_out.company_summary, icp_criteria="CIO at Indian banks"
))

print("3. Strategy Agent Decision...")
s_out, _ = run_strategy(StrategyInput(
    campaign_id="cmp_a", campaign_name="BFSI Voice", prospect_name="Rajesh Kumar", prospect_role="CIO",
    company_name="HDFC Bank", prospect_summary=r_out.company_summary, channel_policy="linkedin, email, call"
))
print(f"   -> Action: {s_out.action} via {s_out.channel} ({s_out.reason})")

print("4. Personalising Email (Visual Tier)...")
p_out, _ = run_personalisation(PersonalisationInput(
    campaign_name="BFSI Voice", 
    prospect_name="Rajesh Kumar", 
    prospect_role="CIO", 
    company_name="HDFC Bank",
    research_facts=r_out.company_summary, 
    intent=s_out.intent, 
    channel=s_out.channel,
    personalization_tier="visual"  # <-- NEW: Requesting visual tier
))
print(f"   Tier:    {p_out.personalization_tier}")
print(f"   Subject: {p_out.subject}")
print(f"   Visual:  {p_out.visual_asset}")

print("5. Fact-Checking Draft (Grounding Check)...")
g_out, _ = check_grounding(
    message=p_out.body,
    sources=[{"id": "res_1", "content": r_out.company_summary}]
)
print(f"   -> All Supported: {g_out.all_supported}")
print(f"   -> Risky Topics:  {g_out.risky_topics}")
print("\n🎉 FULL SDR AGENT PIPELINE PASSED SUCCESSFULLY!")
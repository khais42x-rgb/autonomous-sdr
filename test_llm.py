from backend.agents.research import ResearchInput, run as run_research
from backend.agents.fitment import FitmentInput, run as run_fitment

print("1. Running Research Agent...")
r_input = ResearchInput(
    prospect_name="Rajesh Kumar",
    prospect_role="CIO",
    company_name="HDFC Bank",
    company_domain="hdfcbank.com",
    raw_notes="HDFC Bank is expanding digital customer support and voice collections across India."
)
r_output, r_meta = run_research(r_input)

print("\n--- RESEARCH RESULT ---")
print(f"Summary: {r_output.company_summary}")
print(f"Pains:   {r_output.likely_pains}")

print("\n2. Running Fitment Agent...")
f_input = FitmentInput(
    campaign_id="cmp_bfsi",
    campaign_name="BFSI Voice Agents",
    prospect_name=r_input.prospect_name,
    prospect_role=r_input.prospect_role,
    company_name=r_input.company_name,
    research_summary=r_output.company_summary,
    icp_criteria="Target CIO/CDO at Indian banks or insurers needing Indic-language voice automation."
)
f_output, f_meta = run_fitment(f_input)

print("\n--- FITMENT RESULT ---")
print(f"Verdict: {f_output.verdict} (Score: {f_output.score}/100)")
print(f"Reason:  {f_output.reason}")
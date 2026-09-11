"""BWC Sales Intelligence Streamlit application."""

from html import escape

import streamlit as st

from config import settings
from crm import accounts_csv, contact_template_csv, contacts_csv, import_accounts_csv, import_contacts_csv, PIPELINE_STATUSES, list_accounts, list_contacts, list_prospects, save_account, save_contact, save_prospect
from data import Confidence, Opportunity, TriState
from local_ai import check_ollama, diagnose_ollama, generate_sales_intelligence, ollama_enabled, test_generation
from research import research_company
from sales_action import build_sales_action
from scoring import qualify_company
from utils import normalize_company_name, score_color


def _assessment_value(result, label, value):
    if value.value != "UNKNOWN":
        return value.value
    if result.research_performed:
        return "NO PUBLIC EVIDENCE"
    return "INSUFFICIENT DATA"

st.set_page_config(page_title=settings.app_title, page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;700&display=swap');
:root { --ink:#14242d; --muted:#60727c; --line:#dbe5e5; --mint:#d8f0e7; --teal:#167c70; --cream:#f7f8f3; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
.stApp { background: radial-gradient(circle at 88% 4%, #e3f4ee 0, #f7f8f3 34%, #f7f8f3 100%); }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
.hero { padding: 1.5rem 0 1rem; border-bottom: 1px solid var(--line); margin-bottom: 1.25rem; }
.eyebrow { color: var(--teal); font-size: .75rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
.hero h1 { font-size: 2.25rem; margin: .25rem 0 .35rem; }
.hero p { color: var(--muted); margin: 0; max-width: 720px; }
.metric { background: white; border: 1px solid var(--line); border-radius: 8px; padding: 1rem 1.1rem; min-height: 108px; }
.metric-label { color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }
.metric-value { font-family: 'Space Grotesk'; font-size: 1.45rem; font-weight: 700; margin-top: .55rem; }
.score { background: var(--ink); color: white; border-radius: 10px; padding: 1.25rem; text-align: center; }
.score-number { color: #a9ecd5; font-family: 'Space Grotesk'; font-size: 3.2rem; line-height: 1; font-weight: 700; }
.score-caption { color: #c4d4d5; font-size: .78rem; margin-top: .35rem; }
.evidence { border-left: 3px solid var(--teal); padding: .25rem 0 .25rem .8rem; margin: .75rem 0; }
.evidence p { margin: 0 0 .25rem; }
.source { color: var(--teal); font-size: .82rem; }
.unknown { color: var(--muted); }
.brief { background: #14242d; color: white; border-radius: 10px; padding: 1.2rem 1.35rem; margin: 1rem 0 1.35rem; }
.brief-title { color: #a9ecd5; font-family: 'Space Grotesk'; font-size: 1.1rem; font-weight: 700; }
.brief-why { color: #d7e3e3; margin-top: .85rem; }
div[data-testid="stSidebar"] { background: #eef4f0; border-right: 1px solid var(--line); }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### BWC Intelligence")
    st.caption("Prospect qualification workspace")
    st.divider()
    st.markdown("**Research settings**")
    st.selectbox("Provider", [settings.research_provider.title()], disabled=True)
    st.selectbox("Depth", [settings.research_depth.upper()], disabled=True)
    st.checkbox("Show evidence", value=True)
    st.checkbox("Include unknown signals", value=True)
    st.divider()
    st.markdown("### Local AI")
    st.markdown(f"Mode: {'Disabled' if settings.ai_mode == 'none' else 'Ollama'}")
    st.markdown(f"Model: {settings.ollama_model}")
    st.markdown(f"Endpoint: {settings.ollama_url}")
    if settings.ai_mode == "ollama":
        diagnostic = diagnose_ollama()
        env_label = diagnostic.get("environment", "remote")
        status_label = "Connected" if diagnostic.get("reachable") and diagnostic.get("model_available") else "Not reachable"
        st.markdown(f"Connection: {status_label}")
        st.markdown(f"Environment: {env_label}")
        if diagnostic.get("environment") == "private":
            st.caption("This private/local endpoint may be reachable only on your local network or VPN. A GitHub Codespace usually cannot reach a phone hotspot IP directly.")
    if st.button("Test Ollama Connection", key="test_ollama_connection"):
        health = check_ollama()
        if health.get("available") and health.get("model_available"):
            st.success("✓ Ollama connected")
        else:
            st.warning("⚠ Ollama unavailable")
        st.caption(health.get("message", "Local AI unavailable"))
        if health.get("model_available"):
            st.caption("Configured model exists: yes")
        else:
            st.caption("Configured model exists: no")
    if settings.ai_mode == "ollama":
        if st.button("Test Generation", key="test_generation_button"):
            tiny = test_generation("Reply with exactly: BWC Ollama connection working.")
            if tiny.get("success"):
                st.success("✓ Tiny generation succeeded")
            else:
                st.warning("⚠ Tiny generation failed")
            st.caption(tiny.get("message", "Generation test not run"))
            if tiny.get("response"):
                st.code(tiny.get("response"))
            st.caption(f"Elapsed: {tiny.get('elapsed', 0.0):.3f}s")
    st.divider()
    st.caption("Evidence is directional and should be verified before outreach.")

st.markdown('<div class="hero"><div class="eyebrow">Brainwave Consulting</div><h1>BWC Sales Intelligence</h1><p>Turn a company name into a transparent, evidence-led qualification brief for PLM, engineering data, MSDS, and formulation conversations.</p></div>', unsafe_allow_html=True)

provided_evidence = ""

with st.form("research_form"):
    search_col, button_col = st.columns([5, 1], vertical_alignment="bottom")
    with search_col:
        company_name = st.text_input("Company to research", placeholder="Try Tata Motors", label_visibility="visible")
        provided_evidence = st.text_area(
            "Optional research URLs or notes",
            placeholder="Paste Google/Brave/DuckDuckGo URLs, one per line. Add a note after a | if useful.",
            height=90,
        )
    with button_col:
        submitted = st.form_submit_button("Research", type="primary", width="stretch")

if submitted:
    if not company_name.strip():
        st.warning("Enter a company name to begin research.")
        st.stop()
    with st.status(f"Researching {company_name.strip()}...", expanded=True) as research_status:
        try:
            st.write("Resolving company identity and planning focused research")
            result = research_company(company_name, provided_evidence=provided_evidence)
            st.write(f"Collected {result.useful_result_count} useful evidence records")
            qualification = qualify_company(result)
            qualification.sales_action_layer = build_sales_action(result, qualification)
            save_account(
                company=result.overview.name,
                website=result.overview.website,
                industry=result.overview.industry,
                research_quality=result.research_quality_score,
                sales_score=qualification.sales_score,
                recommendation=qualification.sales_action,
                solution=qualification.solution_recommendation,
                research_status=qualification.sales_action_layer.research_confidence,
                next_action=qualification.sales_action_layer.next_action,
                research_updated_at=result.research_audit.get("started_at", ""),
            )
            st.session_state["result"] = result
            st.session_state["qualification"] = qualification
            st.session_state["normalized_name"] = normalize_company_name(company_name)
            st.session_state.pop("ai_sales_intelligence", None)
            if settings.ai_mode == "ollama":
                health = check_ollama()
                if health.get("available") and health.get("model_available"):
                    ai_result = generate_sales_intelligence(result, qualification)
                    st.session_state["ai_sales_intelligence"] = ai_result
                else:
                    st.session_state["ai_sales_intelligence"] = {}
            else:
                st.session_state["ai_sales_intelligence"] = {}
            research_status.update(label="Research complete", state="complete", expanded=False)
        except Exception as error:
            research_status.update(label="Research failed", state="error", expanded=True)
            st.error("Research could not be completed. Please try again.")
            st.caption(f"Specific error: {error}")

with st.expander("CRM accounts and contacts", expanded="result" not in st.session_state):
    export_col, template_col = st.columns(2)
    with export_col:
        st.download_button("Export accounts CSV", data=accounts_csv(), file_name="bwc_accounts.csv", mime="text/csv", width="stretch")
        st.download_button("Export contacts CSV", data=contacts_csv(), file_name="bwc_contacts.csv", mime="text/csv", width="stretch")
    with template_col:
        st.download_button("Download contact template", data=contact_template_csv(), file_name="bwc_contacts_template.csv", mime="text/csv", width="stretch")
        st.caption("Edit the CSV in Excel or Sheets, then use it as a clean reference for entering contacts.")
    st.markdown("**Import CSV data**")
    import_accounts_col, import_contacts_col = st.columns(2)
    with import_accounts_col:
        accounts_file = st.file_uploader("Accounts CSV", type="csv", key="accounts_csv_upload")
        if accounts_file is not None and st.button("Import accounts", key="import_accounts_button"):
            imported, errors = import_accounts_csv(accounts_file.getvalue())
            if imported:
                st.success(f"Imported or updated {imported} account(s).")
            for error in errors:
                st.warning(error)
    with import_contacts_col:
        contacts_file = st.file_uploader("Contacts CSV", type="csv", key="contacts_csv_upload")
        if contacts_file is not None and st.button("Import contacts", key="import_contacts_button"):
            imported, errors = import_contacts_csv(contacts_file.getvalue())
            if imported:
                st.success(f"Imported or updated {imported} contact(s).")
            for error in errors:
                st.warning(error)
    account_col, contact_col = st.columns(2)
    with account_col:
        st.markdown("**Add or update account**")
        with st.form("crm_account_form"):
            account_name = st.text_input("Account name", placeholder="Tata Motors")
            account_website = st.text_input("Website", placeholder="https://www.example.com")
            account_industry = st.text_input("Industry", placeholder="Automotive manufacturing")
            account_notes = st.text_area("Account notes", placeholder="Context for this account.")
            save_account_button = st.form_submit_button("Save account")
            if save_account_button:
                if not account_name.strip():
                    st.warning("Enter an account name.")
                else:
                    save_account(company=account_name, website=account_website, industry=account_industry, notes=account_notes)
                    st.success(f"Saved {account_name.strip()}.")
    with contact_col:
        st.markdown("**Add or update contact**")
        accounts = list_accounts()
        account_names = [item.get("company", "") for item in accounts if item.get("company")]
        if account_names:
            with st.form("crm_contact_form"):
                contact_account = st.selectbox("Account", account_names)
                contact_name = st.text_input("Contact name", placeholder="Jane Smith")
                contact_role = st.text_input("Role", placeholder="Engineering Systems Manager")
                contact_email = st.text_input("Email", placeholder="jane.smith@example.com")
                contact_phone = st.text_input("Phone", placeholder="+1 555 0100")
                contact_notes = st.text_area("Contact notes", placeholder="Role context or outreach notes.")
                save_contact_button = st.form_submit_button("Save contact")
                if save_contact_button:
                    if not contact_name.strip():
                        st.warning("Enter a contact name.")
                    else:
                        save_contact(account=contact_account, name=contact_name, role=contact_role, email=contact_email, phone=contact_phone, notes=contact_notes)
                        st.success(f"Saved {contact_name.strip()} to {contact_account}.")
            contacts = list_contacts()
            if contacts:
                st.dataframe([
                    {"Account": item["account"], "Name": item["name"], "Role": item["role"], "Email": item["email"], "Phone": item["phone"]}
                    for item in contacts
                ], width="stretch", hide_index=True)
        else:
            st.caption("Add an account before adding contacts.")

if "result" not in st.session_state:
    st.info("Enter a company above to generate a prospect qualification brief.")
    st.stop()

result = st.session_state["result"]
qualification = st.session_state["qualification"]
sales_action = qualification.sales_action_layer
st.caption(f"Normalized search: `{st.session_state['normalized_name']}` · Provider: {result.provider_name}")

st.markdown(f"## {result.overview.name}")
st.markdown("### Sales action")
st.markdown(
    f'<div class="brief"><div class="brief-title">PRIORITY: {escape(sales_action.sales_readiness)}</div>'
    f'<div class="brief-why"><strong>RECOMMENDED SOLUTION:</strong> {escape(qualification.solution_recommendation)} &nbsp; '
    f'<strong>GREENFIELD POTENTIAL:</strong> {escape(qualification.greenfield_opportunity.value)} &nbsp; '
    f'<strong>BEST PERSONA:</strong> {escape(sales_action.primary_target)} &nbsp; '
    f'<strong>RESEARCH CONFIDENCE:</strong> {escape(sales_action.research_confidence)}</div>'
    f'<div class="brief-why"><strong>SALES ANGLE:</strong> {escape(sales_action.sales_angle)}</div>'
    f'<div class="brief-why"><strong>NEXT ACTION:</strong> {escape(sales_action.next_action)}</div></div>',
    unsafe_allow_html=True,
)
st.markdown("### Sales brief")
why = qualification.solution_explanation
if qualification.score_reasons:
    why += " " + qualification.score_reasons[0]
st.markdown(
    f'<div class="brief"><div class="brief-title">TARGET: {escape(qualification.solution_recommendation)}</div>'
    f'<div class="brief-why"><strong>PRIORITY:</strong> {escape(qualification.sales_action)} &nbsp; '
    f'<strong>GREENFIELD PLM:</strong> {escape(qualification.greenfield_opportunity.value)} &nbsp; '
    f'<strong>PLM SERVICES:</strong> {escape(qualification.plm_services_opportunity.value)} &nbsp; '
    f'<strong>MSDS:</strong> {escape(qualification.msds_opportunity.value)} &nbsp; '
    f'<strong>FORMULATION:</strong> {escape(qualification.formulation_opportunity.value)}</div>'
    f'<div class="brief-why"><strong>WHY:</strong> {escape(why)}</div></div>',
    unsafe_allow_html=True,
)

st.markdown("### CRM prospect record")
crm_col, saved_col = st.columns([1.4, 1])
with crm_col:
    with st.form("crm_prospect_form"):
        owner = st.text_input("Owner", placeholder="Akshar")
        pipeline_status = st.selectbox("Pipeline status", PIPELINE_STATUSES, index=0)
        follow_up_date = st.date_input("Follow-up date", value=None)
        notes = st.text_area("CRM notes", placeholder="Add qualification notes or outreach context.")
        save_crm = st.form_submit_button("Save prospect", type="primary")
        if save_crm:
            saved = save_prospect(
                company=result.overview.name,
                industry=result.overview.industry,
                website=result.overview.website,
                sales_score=qualification.sales_score,
                research_quality=result.research_quality_score,
                recommendation=qualification.sales_action,
                solution=qualification.solution_recommendation,
                primary_persona=sales_action.primary_target,
                secondary_personas=sales_action.secondary_targets,
                next_action=sales_action.next_action,
                research_status=sales_action.research_confidence,
                notes=notes,
                owner=owner,
                pipeline_status=pipeline_status,
                follow_up_date=follow_up_date.isoformat() if follow_up_date else "",
            )
            st.success(f"Saved {saved['company']} to the local CRM.")
with saved_col:
    st.markdown("**Saved prospects**")
    prospects = list_prospects()
    if prospects:
        st.dataframe([
            {"Company": item.get("company", ""), "Status": item.get("pipeline_status", "NEW"), "Sales score": item.get("sales_score", ""), "Owner": item.get("owner", "")}
            for item in prospects
        ], width="stretch", hide_index=True)
    else:
        st.caption("No prospects saved yet.")

score_col, quality_col, action_col, solution_col = st.columns([1.0, 1.0, 1.4, 1.4])
with score_col:
    st.markdown(f'<div class="score"><div class="score-number">{qualification.sales_score}</div><div class="score-caption">SALES SCORE / 100</div></div>', unsafe_allow_html=True)
with quality_col:
    st.markdown(f'<div class="metric"><div class="metric-label">Research quality</div><div class="metric-value">{result.research_quality_score}/100</div></div>', unsafe_allow_html=True)
with action_col:
    st.markdown(f'<div class="metric"><div class="metric-label">Recommended sales action</div><div class="metric-value">{escape(qualification.sales_action)}</div></div>', unsafe_allow_html=True)
with solution_col:
    st.markdown(f'<div class="metric"><div class="metric-label">Recommended BWC solution</div><div class="metric-value">{escape(qualification.solution_recommendation)}</div></div>', unsafe_allow_html=True)

overview_col, opportunity_col = st.columns([1.15, 1])
with overview_col:
    st.markdown("### Company overview")
    st.table({"Field": ["Website", "Industry", "Headquarters", "Employees"], "Value": [result.overview.website or "Unknown", result.overview.industry, result.overview.headquarters, result.overview.employee_range]})
    st.write(result.overview.description)
with opportunity_col:
    st.markdown("### BWC opportunity")
    st.table({"Area": ["PLM/PDM", "PLM services", "MSDS", "Formulation", "Overall"], "Opportunity": [qualification.greenfield_opportunity.value, qualification.plm_services_opportunity.value, qualification.msds_opportunity.value, qualification.formulation_opportunity.value, qualification.overall_opportunity.value]})
    st.info(qualification.sales_action_explanation)

st.markdown("### Manufacturing & engineering")
status_cols = st.columns(4)
for column, label, value in zip(status_cols, ["Manufacturing", "Engineering / R&D", "Product development", "Physical products"], [result.manufacturing, result.engineering_rd, result.product_development, result.physical_products]):
    with column:
        confidence = result.assessment_confidence.get(label, "LOW")
        st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{_assessment_value(result, label, value)}</div></div>', unsafe_allow_html=True)
        st.caption(f"Confidence: {confidence}")

st.markdown("### Existing PLM / PDM landscape")
landscape_rows = [{
    "System": item.system,
    "Status": item.status.value,
    "Confidence": item.confidence,
    "Evidence": item.evidence,
    "Date": item.source_date,
    "Source": f"[{item.source_name}]({item.source_url})" if item.source_url else "No reliable public evidence found",
} for item in result.plm_landscape]
if landscape_rows:
    st.table(landscape_rows)
else:
    st.info("No reliable public evidence found.")
st.caption("NO PUBLIC EVIDENCE means the system was not identified in the sources reviewed; it does not mean the company does not use it.")

plm_analysis = result.research_analysis.get("plm", {})
if plm_analysis:
    st.markdown("### PLM evidence synthesis")
    st.table({
        "Conclusion": [plm_analysis.get("value", "UNCERTAIN")],
        "Status": [plm_analysis.get("status", "INSUFFICIENT DATA")],
        "Confidence": [f"{plm_analysis.get('confidence', 0)}/100"],
        "Evidence records": [len(plm_analysis.get("evidence", []))],
        "Reasoning": [plm_analysis.get("reasoning", "No synthesis available.")],
    })

st.markdown("### Search coverage")
coverage_rows = [{"Category": category, "Coverage": status} for category, status in result.coverage.items()]
if coverage_rows:
    st.table(coverage_rows)

if result.gap_fields:
    with st.expander("Why information is unavailable", expanded=False):
        st.caption("These fields triggered targeted second-pass research and still lack reliable direct public evidence.")
        for field in result.gap_fields:
            st.markdown(f"**Field:** {field}")
            st.markdown(f"**Status:** {result.coverage.get(field, 'NO PUBLIC EVIDENCE')}")
            st.markdown("**Research performed:** planned web search, DuckDuckGo, Bing, GDELT, and relevant first-party/public sources")

st.markdown("### Engineering complexity")
complexity = result.engineering_complexity
st.table({"Dimension": ["Product complexity", "Engineering complexity", "Manufacturing complexity", "Product categories", "Multi-site engineering", "Change management"], "Assessment": [complexity.product_complexity.value, complexity.engineering_complexity.value, complexity.manufacturing_complexity.value, complexity.product_categories.value, complexity.multi_site_engineering.value, complexity.change_management.value]})
st.info(complexity.explanation)

st.markdown("### Buying signals")
if result.buying_signals:
    st.table([{"Signal": item.signal, "Date": item.date, "Confidence": item.confidence, "Evidence": item.evidence or item.why_it_matters, "BWC action": item.recommended_action, "Source": f"[{item.source_name}]({item.source_url})"} for item in result.buying_signals])
else:
    st.info("None identified in the recent public sources searched.")

st.markdown("### Hiring signals")
if result.hiring_signals:
    st.table([{"Signal": item.signal, "Technology": item.technology, "Recency": item.recency, "Confidence": item.confidence, "Date": item.date, "Source": f"[{item.source_name}]({item.source_url})"} for item in result.hiring_signals])
else:
    st.info("None identified in the public hiring sources searched.")

st.markdown("### Pain-point indicators")
if result.pain_points:
    st.table([{"Indicator": item.statement, "Source": f"[{item.source_name}]({item.source_url})"} for item in result.pain_points])
else:
    st.info("No public evidence found.")

st.markdown("### Technology signals")
signal_rows = [{"Signal": signal, "Assessment": confidence.value} for signal, confidence in result.technology_signals.items()]
st.dataframe(signal_rows, width="stretch", hide_index=True)

with st.expander("Score breakdown and recommendation reasoning", expanded=True):
    if qualification.score_breakdown:
        st.table([{"Category": category, "Points": points} for category, points in qualification.score_breakdown.items()])
    if qualification.score_reasons:
        for reason in qualification.score_reasons:
            st.markdown(f"- {reason}")
    else:
        st.markdown('<span class="unknown">No positive scoring evidence was identified.</span>', unsafe_allow_html=True)

with st.expander("Research performed", expanded=False):
    research_log = result.research_performed
    audit = result.research_audit
    st.write(f"**Research mode:** {'GEMINI + GOOGLE SEARCH' if settings.gemini_api_key else 'NORMAL'}")
    st.write(f"**Research tasks:** {audit.get('first_pass_tasks', 0)} first pass / {audit.get('second_pass_tasks', 0)} second pass")
    st.write(f"**Planned queries:** {len(audit.get('initial_queries', []))} initial / {len(audit.get('followup_queries', []))} follow-up")
    st.write(f"**Useful sources:** {audit.get('useful_sources', result.useful_result_count)}")
    st.write(f"**Unique domains:** {audit.get('unique_domains', 0)}")
    st.write(f"**Pages considered/read:** {audit.get('pages_considered', 0)} / {audit.get('pages_read', 0)}")
    st.write(f"**Reader failures:** {audit.get('reader_failures', 0)}")
    st.write(f"**SearXNG:** {'enabled' if audit.get('searxng_enabled') else 'disabled/fallback'}")
    st.write(f"**Jina Reader:** {'enabled' if audit.get('jina_enabled') else 'disabled/fallback'}")
    st.write("**Specialized modules:** " + ", ".join(audit.get("specialized_modules", [])))
    st.write(f"**Wayback used:** {audit.get('wayback_used', False)} · **GitHub used:** {audit.get('github_used', False)}")
    st.write(f"**Research quality:** {result.research_quality_score}/100")
    if audit.get("provider_stats"):
        st.write("**Free-provider diagnostics:**")
        st.table([{"Provider": name, **stats} for name, stats in audit["provider_stats"].items()])
    st.write("**Categories searched:** " + ", ".join(audit.get("categories_searched", [])))
    st.metric("Useful results", result.useful_result_count)
    st.write(f"**Queries searched:** {len(research_log.get('Queries searched', []))}")
    for query in research_log.get("Queries searched", []):
        st.markdown(f"- `{query}`")
    st.write(f"**Sources checked:** {len(research_log.get('Sources checked', []))}")
    st.write(", ".join(research_log.get("Sources checked", [])) or "None")
    st.write("**Technology platforms checked:**")
    st.write(", ".join(research_log.get("Technology platforms checked", [])) or "None")
    st.write(f"**Job searches performed:** {len(research_log.get('Job searches performed', []))}")

with st.expander("Best contacts to target", expanded=False):
    st.markdown(f"**Primary target:** {sales_action.primary_target}")
    if sales_action.persona_reasons.get(sales_action.primary_target):
        st.caption(sales_action.persona_reasons[sales_action.primary_target])
    if sales_action.secondary_targets:
        st.markdown("**Secondary targets**")
        for contact in sales_action.secondary_targets:
            st.markdown(f"- {contact}: {sales_action.persona_reasons.get(contact, 'Relevant to the qualified opportunity.')}")
    if sales_action.supporting_target:
        st.markdown(f"**Supporting target:** {sales_action.supporting_target}")

st.markdown("### Recommended sales angle")
st.info(sales_action.sales_angle)

st.markdown("### Why this company?")
for talking_point in sales_action.talking_points:
    st.markdown(f"- {talking_point}")

st.markdown("### Discovery questions")
if sales_action.discovery_questions:
    for question in sales_action.discovery_questions:
        st.markdown(f"- {question}")
else:
    st.info("No solution-specific discovery questions generated until a clearer opportunity is identified.")

st.markdown("### Outreach preparation")
outreach_tabs = st.tabs(["Opening message", "Call opener", "Email draft", "LinkedIn angle"])
with outreach_tabs[0]:
    st.write(sales_action.opening_message)
with outreach_tabs[1]:
    st.write(sales_action.call_opener)
with outreach_tabs[2]:
    st.text_area("Email draft", value=sales_action.email_draft, height=220, label_visibility="collapsed")
with outreach_tabs[3]:
    st.write(sales_action.linkedin_angle)

readiness_col, matrix_col = st.columns(2)
with readiness_col:
    st.markdown("### Research confidence")
    st.markdown(f'<div class="metric"><div class="metric-label">Confidence</div><div class="metric-value">{escape(sales_action.research_confidence)}</div></div>', unsafe_allow_html=True)
    st.caption(sales_action.confidence_explanation)
with matrix_col:
    st.markdown("### Sales readiness")
    st.markdown(f'<div class="metric"><div class="metric-label">Readiness</div><div class="metric-value">{escape(sales_action.sales_readiness)}</div></div>', unsafe_allow_html=True)
    st.caption(sales_action.readiness_explanation)

st.markdown("### Priority matrix")
st.info(sales_action.priority_matrix)

st.markdown("### Local AI Sales Intelligence")
health = check_ollama() if settings.ai_mode == "ollama" else {"available": False, "model_available": False, "message": "Local AI is disabled. Set BWC_AI_MODE=ollama to enable Ollama.", "environment": "remote"}
if settings.ai_mode == "none":
    st.info("Local AI is disabled.\nSet BWC_AI_MODE=ollama to enable Ollama.")
elif not health.get("available") or not health.get("model_available"):
    diagnostic = diagnose_ollama()
    st.warning("Local AI unavailable — showing deterministic BWC qualification only.")
    st.caption(diagnostic.get("message", "Local AI unavailable"))
    env = diagnostic.get("environment", "remote")
    if env == "private":
        st.caption("Your Ollama endpoint is on a private network. A GitHub Codespace cannot normally reach your phone's hotspot IP directly. Use a secure VPN/private tunnel or run Ollama somewhere reachable by the Codespace.")
    if diagnostic.get("error_type") == "connection_refused":
        st.caption("Connection refused: Ollama server is not listening on the configured address/port.")
else:
    diagnostic = diagnose_ollama()
    st.success("Local AI available")
    st.caption(f"Mode: Ollama | Model: {settings.ollama_model} | Endpoint: configured | Connection: {'Connected' if diagnostic.get('reachable') else 'Not reachable'} | Environment: {diagnostic.get('environment', 'remote')}")
    ai = st.session_state.get("ai_sales_intelligence")
    if not isinstance(ai, dict) or not ai:
        ai = generate_sales_intelligence(result, qualification)
        st.session_state["ai_sales_intelligence"] = ai
    if st.button("Regenerate AI Analysis", key="regenerate_ai_analysis"):
        ai = generate_sales_intelligence(result, qualification)
        st.session_state["ai_sales_intelligence"] = ai
    # Render key fields in a compact, non-breaking section.
    st.markdown("**Executive Summary**")
    st.markdown(ai.get("executive_summary") or "")
    st.markdown("**Why This Company May Be Worth Calling**")
    st.markdown(ai.get("why_worth_calling") or "")
    st.markdown("**PLM/PDM Opportunity**")
    st.markdown(ai.get("plm_pdm_opportunity") or "")
    st.markdown("**MSDS Opportunity**")
    st.markdown(ai.get("msds_opportunity") or "")
    st.markdown("**Formulation Opportunity**")
    st.markdown(ai.get("formulation_opportunity") or "")
    st.markdown("**Greenfield Assessment**")
    st.markdown(ai.get("greenfield_assessment") or "")
    st.markdown("**Recommended Persona**")
    st.markdown(ai.get("recommended_persona") or "")
    st.markdown("**Recommended Call Angle**")
    st.markdown(ai.get("recommended_call_angle") or "")
    st.markdown("**Recommended Next Step**")
    st.markdown(ai.get("recommended_next_step") or "")
    st.markdown("**AI Confidence**")
    st.markdown(ai.get("confidence") or "")
    with st.expander("Key evidence", expanded=False):
        for item in ai.get("key_evidence") or []:
            st.markdown(f"- {item}")
    with st.expander("Risks / Unknowns", expanded=False):
        for item in ai.get("risks_or_unknowns") or []:
            st.markdown(f"- {item}")
    with st.expander("Likely pain points", expanded=False):
        for item in ai.get("likely_pain_points") or []:
            st.markdown(f"- {item}")
    with st.expander("Discovery questions", expanded=False):
        for item in ai.get("discovery_questions") or []:
            st.markdown(f"- {item}")
    with st.expander("Sales talking points", expanded=False):
        for item in ai.get("sales_talking_points") or []:
            st.markdown(f"- {item}")

with st.expander("Evidence and sources", expanded=True):
    if result.research_note:
        st.caption(result.research_note)
    if result.evidence:
        for item in result.evidence:
            st.markdown(f'<div class="evidence"><p>{escape(item.statement)}</p><span class="source">{escape(item.fact_or_inference)} · {escape(item.confidence)} · {escape(item.source_type)}</span><br><a class="source" href="{escape(item.source_url)}" target="_blank">Source: {escape(item.source_name)}</a></div>', unsafe_allow_html=True)
    else:
        st.warning("No reliable public evidence found.")
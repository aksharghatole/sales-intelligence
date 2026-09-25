# BWC Sales Intelligence

Sales intelligence and prospect qualification for Brainwave Consulting. Point it at a company and it researches public sources, scores the fit against BWC's PLM/PDM, ENOVIA, 3DEXPERIENCE, engineering data, MSDS/SDS, and formulation opportunities, then produces a sales brief with recommended next actions.

![BWC Sales Intelligence screenshot](docs/screenshot.png)

## Features

- **Automated company research** — resolves identity via Wikidata, then runs concurrent research across the official site, news, filings, jobs, vendor ecosystems, and GitHub activity
- **Deterministic scoring** — 100-point qualification model with greenfield and services-opportunity rules
- **Sales briefs** — role-based personas, sales hypotheses, discovery questions, and outreach drafts
- **Source coverage & gaps** — every result reports what was found, what was read, and explicit `NO PUBLIC EVIDENCE` gaps
- **Lightweight CRM** — local accounts/contacts with CSV import/export and pipeline tracking
- **Optional local AI** — enrich results with a local Ollama model; the deterministic score remains authoritative
- **Optional SearXNG** — broad discovery via a self-hosted meta-search engine

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py

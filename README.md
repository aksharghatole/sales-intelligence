# BWC Sales Intelligence

Sales intelligence and prospect qualification for Brainwave Consulting. The app evaluates a company against BWC's PLM/PDM, ENOVIA, 3DEXPERIENCE, engineering data, MSDS/SDS, and formulation opportunities.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

The default provider is `live`. It resolves the company through Wikidata, asks Gemini for a compact research plan when `GEMINI_API_KEY` is configured, runs independent research areas concurrently, deduplicates and ranks raw results, reads the highest-value pages through Jina Reader, and performs targeted follow-up research for detected gaps. Optional SearXNG provides broad discovery; Gemini uses Google Search grounding when available. Existing DuckDuckGo, Bing, Brave, GDELT, Google News RSS, Wikipedia, Wikidata, and first-party fallbacks remain available. Results include source coverage, research quality, measured duration, page-read counts, and explicit `NO PUBLIC EVIDENCE` gaps.

DuckDuckGo uses the HTML endpoint, Lite endpoint, and free Instant Answer API fallback. Free providers use bounded retries, canonical URL normalization, tolerant parsing, per-provider diagnostics, and a per-run circuit breaker so an unavailable endpoint is not retried for every query. Healthy providers continue contributing evidence.

## Configuration

Environment variables are loaded from `.env`:

- `BWC_RESEARCH_PROVIDER`: provider name, currently `live` or `demo`
- `BWC_APP_TITLE`: optional browser/page title override
- `BWC_REQUEST_TIMEOUT`: HTTP timeout for future providers, in seconds
- `BWC_FREE_PROVIDER_TIMEOUT`: strict timeout for each free search request, default `3` seconds
- `BWC_PROVIDER_RETRY_ATTEMPTS`: retry count for free providers, default `1`
- `BWC_RESEARCH_CONCURRENCY`: maximum concurrent research tasks, default `8`
- `BWC_RESEARCH_CACHE_TTL`: in-process cache lifetime in seconds, default `900`
- `BWC_RESEARCH_DEPTH`: `quick`, `normal`, or `deep`; default `normal`
- `BRAVE_SEARCH_API_KEY`: optional Brave Search fallback key
- `SEARXNG_ENABLED`: enable SearXNG discovery, default `false`
- `SEARXNG_URL`: SearXNG base URL, default `http://localhost:8080`
- `JINA_ENABLED`: enable ranked page reading, default `true`
- `JINA_READER_URL`: Jina Reader base URL, default `https://r.jina.ai`
- `JINA_API_KEY`: optional Jina API key
- `DIRECT_READER_ENABLED`: use direct public-page extraction when Jina fails, default `true`
- `MAX_RESEARCH_ROUNDS`: research budget, default `3`
- `MAX_SEARCH_QUERIES_PER_ROUND`: query budget setting, default `10`
- `MAX_RESULTS_PER_QUERY`: result limit per provider query, default `10`
- `MAX_PAGES_TO_READ`: maximum ranked pages sent to Jina, default `15`
- `MAX_FOLLOWUP_QUERIES`: follow-up query budget setting, default `10`
- `RESEARCH_DEBUG`: enable additional provider audit metadata, default `false`
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`: optional Google AI Studio key for Gemini + Google Search grounding
- `GEMINI_MODEL`: optional Gemini model, default `gemini-2.0-flash`

## Local AI with Ollama

BWC also supports an optional local AI enrichment layer powered by a locally running Ollama server. This is entirely optional and disabled by default. Set the environment variables in `.env` or your shell before launching Streamlit:

```env
BWC_AI_MODE=ollama
BWC_OLLAMA_URL=http://127.0.0.1:11434
BWC_OLLAMA_MODEL=llama3.2:1b
BWC_AI_TIMEOUT=60
BWC_AI_MAX_INPUT_CHARS=30000
```

The deterministic BWC research and scoring flow remains authoritative. Ollama does not replace the deterministic qualification; it enriches it with explanation, evidence interpretation, and sales intelligence. If Ollama is unavailable, the application keeps showing the normal research and scoring results. For a local Ollama server, use `BWC_OLLAMA_URL=http://127.0.0.1:11434`. For a different reachable host, use `BWC_OLLAMA_URL=http://HOST:11434`.

When `BWC_AI_MODE=none`, the app works exactly as before and requires no local AI service. When `BWC_AI_MODE=ollama`, the UI will show the optional AI section after the deterministic score and sales brief if the configured model is reachable and available.

No API keys are stored in source code. Add provider credentials to `.env` or Streamlit secrets.

## Optional SearXNG

The Streamlit app does not require SearXNG. To run the included local setup:

```bash
docker compose up -d searxng valkey
```

Then set `SEARXNG_ENABLED=true` and `SEARXNG_URL=http://localhost:8080` in `.env`, and restart Streamlit. If SearXNG is off or unavailable, the app continues with the existing public providers. Jina Reader is also failure-isolated: search snippets remain available when a page cannot be read.

## Lightweight CRM

The current app includes a local account and contact CRM. Add accounts with a website, industry, and notes; add contacts linked to an account with name, role, email, phone, and notes. Research automatically creates or enriches the researched account with scores, recommendation, solution, research confidence, and next action. Use the CSV download buttons to export accounts or contacts, and the upload controls to import edited CSV files. Contacts must reference an existing account; invalid rows are reported without stopping the import. Imports upsert by account/contact name. Use **Save prospect** to additionally store pipeline status, owner, follow-up date, and outreach notes. Records are stored in `BWC_CRM_PATH` (default `crm_data.json`), which is ignored by Git. This is intentionally local and dependency-free.

## Specialized research

After identity resolution, the live provider runs bounded specialist searches for the official company site, vendor/customer ecosystems, jobs, annual reports and investor presentations, recent news, public filings, and relevant GitHub activity. Wayback Machine is used only as a targeted historical fallback when current PLM evidence is weak. Results are ranked, deduplicated, and read through Jina before the existing evidence synthesis and follow-up loop. Historical evidence is labeled separately and does not become a current technology fact.

## Project layout

- `app.py`: Streamlit presentation layer
- `data.py`: dataclasses and enums for research and qualification results
- `research.py`: provider interfaces and public-source signal extraction
- `scoring.py`: 100-point qualification, greenfield, and services opportunity rules
- `sales_action.py`: role-based personas, sales hypotheses, discovery questions, and outreach drafts
- `utils.py`: normalization and formatting helpers
- `config.py`: environment-backed configuration
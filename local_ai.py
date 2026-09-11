"""Local AI / Ollama enrichment for BWC Sales Intelligence.

This module intentionally keeps all Ollama-specific HTTP and response handling
in one place so the existing research and scoring pipeline stays deterministic.
"""

import json
import logging
import requests
from typing import Any

from config import settings
from data import Qualification, ResearchResult, SalesAction


SYSTEM_PROMPT = (
    "You are the local AI sales intelligence assistant for Brainwave Consulting. "
    "You are assisting a B2B pre-sales/business-development workflow. "
    "Your job is to interpret already-collected public research and the deterministic "
    "BWC qualification output. Never invent company facts. Never claim that a company "
    "uses a particular PLM/PDM/ERP/CAD system unless the supplied evidence supports it. "
    "Clearly distinguish confirmed evidence, reasonable inference, and unknown/unverified "
    "information. Do not change the official BWC score. Do not fabricate contact names, "
    "budgets, projects, technology deployments, or purchasing timelines. Focus on: "
    "PLM/PDM, ENOVIA, 3DEXPERIENCE, engineering data, product lifecycle, CAD, BOM, "
    "engineering change management, MSDS/SDS, formulation management, regulatory workflows, "
    "manufacturing, R&D, quality, and SAP/ERP integration. Give practical sales intelligence. "
    "Return ONLY valid JSON."
)


AI_OUTPUT_SCHEMA = {
    "executive_summary": "",
    "qualification_explanation": "",
    "why_worth_calling": "",
    "plm_pdm_opportunity": "",
    "msds_opportunity": "",
    "formulation_opportunity": "",
    "greenfield_assessment": "",
    "key_evidence": [],
    "risks_or_unknowns": [],
    "likely_pain_points": [],
    "recommended_persona": "",
    "recommended_call_angle": "",
    "discovery_questions": [],
    "sales_talking_points": [],
    "recommended_next_step": "",
    "confidence": "",
}


REQUEST_LOGGER = logging.getLogger("bwc.local_ai")


def ollama_enabled() -> bool:
    """Return True when the application is configured for local Ollama.

    The configuration is environment controlled and copied into Settings.
    """
    return settings.ai_mode.lower() == "ollama"


def _ai_disabled_result() -> dict:
    return {
        **AI_OUTPUT_SCHEMA,
        "executive_summary": "Local AI is disabled.",
        "confidence": "Disabled",
    }


def _safe_model_names(models_payload: Any) -> list[str]:
    models: list[str] = []
    try:
        if isinstance(models_payload, dict):
            candidates = models_payload.get("models", [])
            for item in candidates:
                if isinstance(item, dict):
                    name = item.get("name")
                    if name:
                        models.append(name)
                elif isinstance(item, str):
                    models.append(item)
        elif isinstance(models_payload, list):
            for item in models_payload:
                if isinstance(item, dict):
                    name = item.get("name")
                    if name:
                        models.append(name)
                elif isinstance(item, str):
                    models.append(item)
    except Exception:
        pass
    return models


def check_ollama() -> dict:
    """Perform a lightweight health check against Ollama.

    Returns a structured dict so Streamlit/UI code can render a soft message.
    """
    if not ollama_enabled():
        return {
            "available": False,
            "model_available": False,
            "message": "Local AI is disabled. Set BWC_AI_MODE=ollama to enable Ollama.",
        }

    url = f"{settings.ollama_url.rstrip('/')}/api/tags"
    try:
        response = requests.get(url, timeout=min(settings.ai_timeout, 10))
        if response.status_code != 200:
            return {
                "available": False,
                "model_available": False,
                "message": "Ollama is not reachable",
            }
        try:
            payload = response.json()
        except Exception:
            payload = {}
        names = _safe_model_names(payload)
        model_found = settings.ollama_model in names
        if model_found:
            return {
                "available": True,
                "model_available": True,
                "message": "✓ Ollama connected",
            }
        return {
            "available": True,
            "model_available": False,
            "message": "Configured Ollama model was not found",
        }
    except requests.exceptions.ConnectionError:
        return {
            "available": False,
            "model_available": False,
            "message": "Ollama is not reachable",
        }
    except requests.exceptions.Timeout:
        return {
            "available": False,
            "model_available": False,
            "message": "Ollama request timed out",
        }
    except requests.exceptions.RequestException:
        return {
            "available": False,
            "model_available": False,
            "message": "Ollama is not reachable",
        }
    except Exception:
        return {
            "available": False,
            "model_available": False,
            "message": "Local AI unavailable",
        }


def generate_ai(prompt: str, system_prompt: str = "") -> str | None:
    """Generate a text response from the configured local Ollama model.

    Returns None when AI is disabled, not available, or an exception occurs.
    """
    if not ollama_enabled():
        return None

    url = f"{settings.ollama_url.rstrip('/')}/api/generate"
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt[:settings.ai_max_input_chars],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    if system_prompt:
        payload["system"] = system_prompt

    try:
        response = requests.post(url, json=payload, timeout=settings.ai_timeout)
        response.raise_for_status()
        data = response.json()
        return str(data.get("response", "") or "").strip() or None
    except requests.exceptions.Timeout:
        REQUEST_LOGGER.warning("Ollama request timed out")
        return None
    except requests.exceptions.ConnectionError:
        REQUEST_LOGGER.warning("Ollama connection failure")
        return None
    except requests.exceptions.RequestException as exc:
        REQUEST_LOGGER.warning("Ollama request failure: %s", exc.__class__.__name__)
        return None
    except Exception:
        REQUEST_LOGGER.warning("Malformed Ollama response")
        return None


def _shorten(value: str, max_chars: int = 5000) -> str:
    text = (value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _signal_map(signals: dict[str, Any]) -> str:
    if not signals:
        return "No technology signals identified."
    parts = []
    for key, value in signals.items():
        if isinstance(value, str):
            parts.append(f"{key}: {value}")
        else:
            parts.append(f"{key}: {getattr(value, 'value', str(value))}")
    return "; ".join(parts[:12])


def _build_prompt(result: ResearchResult, qualification: Qualification) -> str:
    """Create a compact structured prompt from the existing research and scoring data.

    This intentionally avoids dumping raw Python objects into the prompt.
    """
    sales_action = getattr(qualification, "sales_action_layer", None)
    if sales_action is None:
        sales_action = SalesAction()

    # Build compact structured summary.
    lines = []
    lines.append("COMPANY:")
    lines.append(f"- name: {result.overview.name}")
    lines.append(f"- website: {result.overview.website or 'Unknown'}")
    lines.append(f"- industry: {result.overview.industry or 'Unknown'}")
    lines.append(f"- description: {_shorten(result.overview.description, 500)}")

    lines.append("\nRESEARCH:")
    lines.append(f"- research_quality_score: {result.research_quality_score}")
    lines.append(f"- provider: {result.provider_name}")
    lines.append(f"- useful_result_count: {result.useful_result_count}")
    lines.append(f"- research_note: {_shorten(result.research_note, 500)}")

    evidence = []
    for item in result.evidence[:12]:
        evidence.append(f"{item.statement} | source={item.source_name} | url={item.source_url} | confidence={item.confidence} | type={item.source_type}")
    lines.append(f"- relevant_evidence: {'; '.join(evidence) if evidence else 'No reliable public evidence found.'}")

    lines.append(f"- evidence_sources: {len({item.source_url for item in result.evidence if item.source_url})}")
    lines.append(f"- buying_signals: {', '.join(item.signal for item in result.buying_signals[:8]) or 'None identified'}")
    lines.append(f"- hiring_signals: {', '.join(item.signal for item in result.hiring_signals[:8]) or 'None identified'}")
    lines.append(f"- pain_points: {', '.join(item.statement for item in result.pain_points[:8]) or 'None identified'}")
    lines.append(f"- plm_landscape: {', '.join(f'{item.system}:{item.status.value}' for item in result.plm_landscape[:8]) or 'No reliable public evidence found.'}")
    lines.append(f"- engineering_complexity: product={result.engineering_complexity.product_complexity.value}; engineering={result.engineering_complexity.engineering_complexity.value}; manufacturing={result.engineering_complexity.manufacturing_complexity.value}; change_management={result.engineering_complexity.change_management.value}")
    lines.append(f"- technology_signals: {_signal_map(result.technology_signals)}")
    lines.append(f"- cad_signals: {result.domain_signals.get('CAD', 'UNKNOWN')}")
    lines.append(f"- erp_sap_signals: {result.domain_signals.get('ERP/SAP', 'UNKNOWN')}")

    lines.append("\nQUALIFICATION:")
    lines.append(f"- sales_score: {qualification.sales_score}")
    lines.append(f"- sales_action: {qualification.sales_action}")
    lines.append(f"- recommended_solution: {qualification.solution_recommendation}")
    lines.append(f"- solution_explanation: {_shorten(qualification.solution_explanation, 500)}")
    lines.append(f"- greenfield_opportunity: {qualification.greenfield_opportunity.value}")
    lines.append(f"- plm_services_opportunity: {qualification.plm_services_opportunity.value}")
    lines.append(f"- msds_opportunity: {qualification.msds_opportunity.value}")
    lines.append(f"- formulation_opportunity: {qualification.formulation_opportunity.value}")
    lines.append(f"- score_reasons: {'; '.join(qualification.score_reasons[:6]) if qualification.score_reasons else 'No scoring reasons identified'}")

    lines.append("\nSALES_ACTION:")
    lines.append(f"- primary_persona: {sales_action.primary_target}")
    lines.append(f"- secondary_personas: {', '.join(sales_action.secondary_targets[:6]) or 'None'}")
    lines.append(f"- sales_angle: {_shorten(sales_action.sales_angle, 500)}")
    lines.append(f"- discovery_questions: {'; '.join(sales_action.discovery_questions[:7]) if sales_action.discovery_questions else 'None'}")
    lines.append(f"- talking_points: {'; '.join(sales_action.talking_points[:5]) if sales_action.talking_points else 'None'}")
    lines.append(f"- next_action: {_shorten(sales_action.next_action, 500)}")
    lines.append(f"- research_confidence: {sales_action.research_confidence}")
    lines.append(f"- sales_readiness: {sales_action.sales_readiness}")
    lines.append(f"- priority_matrix: {sales_action.priority_matrix}")

    prompt = "\n".join(lines)
    if len(prompt) > settings.ai_max_input_chars:
        prompt = prompt[:settings.ai_max_input_chars]
    return prompt


def _empty_analysis() -> dict:
    return {
        "executive_summary": "",
        "qualification_explanation": "",
        "why_worth_calling": "",
        "plm_pdm_opportunity": "",
        "msds_opportunity": "",
        "formulation_opportunity": "",
        "greenfield_assessment": "",
        "key_evidence": [],
        "risks_or_unknowns": [],
        "likely_pain_points": [],
        "recommended_persona": "",
        "recommended_call_angle": "",
        "discovery_questions": [],
        "sales_talking_points": [],
        "recommended_next_step": "",
        "confidence": "",
    }


def _validate_ai_json(payload: Any) -> dict:
    template = _empty_analysis()
    if not isinstance(payload, dict):
        return template
    for key in template:
        value = payload.get(key, template[key])
        if isinstance(template[key], list):
            if isinstance(value, list):
                template[key] = value[:20]
            else:
                template[key] = []
        elif isinstance(template[key], str):
            template[key] = str(value or "") if value is not None else ""
    return template


def generate_sales_intelligence(result: ResearchResult, qualification: Qualification) -> dict:
    """Create structured local AI insight for the research result and qualification.

    The deterministic qualification remains authoritative; Ollama enriches it.
    """
    if not ollama_enabled():
        return _empty_analysis()

    # Graceful fallback if no result object was supplied.
    if result is None or qualification is None:
        return _empty_analysis()

    prompt = _build_prompt(result, qualification)
    raw_answer = generate_ai(prompt, SYSTEM_PROMPT)
    if not raw_answer:
        return _empty_analysis()

    try:
        parsed = json.loads(raw_answer.strip())
        return _validate_ai_json(parsed)
    except Exception:
        # Raw JSON fallback: preserve the text itself in the main executive field.
        output = _empty_analysis()
        output["executive_summary"] = raw_answer
        output["confidence"] = "Text fallback"
        return output

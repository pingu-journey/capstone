"""Persist a rule-based research plan before creating dynamic LangGraph workers."""

import logging
from hashlib import sha256

from agents.market_eval import _questions
from agents.tech_summary import company_label
from schemas import Candidate, MarketAnalysis, SEGMENT_KO, TaskSpec, TechSummary
from tools.artifacts import read_json, run_dir, write_json

logger = logging.getLogger(__name__)
MARKET_FIELDS = {
    "market_size": ["market_size"], "growth": ["growth", "demand_drivers"],
    "demand": ["customers"], "regulation": ["policy_risks"],
}


def has_content(value):
    if isinstance(value, list):
        return bool(value) and all(has_content(item) for item in value)
    return bool(value) and "공개 정보 부족" not in str(value)


def valid_market_cache(state):
    startup = state["current_startup"]
    key = f"{startup['country'].upper()}|{startup['segment']}"
    try:
        market = MarketAnalysis.model_validate((state.get("market_cache") or {})[key]).model_dump()
        if market["country"].upper() != startup["country"].upper() or market["segment"] != startup["segment"]:
            return None
        index = read_json(state["evidence_index_ref"]) if state.get("evidence_index_ref") else []
        # A schema-valid cache without original excerpts cannot support quality review.
        available = {entry["source_id"] for entry in index if entry.get("text")}
        if not market["sources"] or any(ref["id"] not in available for ref in market["sources"]):
            return None
        if any(figure["source_id"] not in available or not figure["year"].strip() for figure in market["market_size"]):
            market["market_size"] = []
        return market
    except (KeyError, ValueError, OSError):
        return None


def valid_research_cache(state):
    name = state["current_startup"]["name"]
    cached = (state.get("research_cache") or {}).get(name)
    if not cached or cached.get("candidate") != state["current_startup"]:
        return None
    try:
        tech = TechSummary.model_validate(cached["tech_summary"])
        if tech.insufficient or not cached["references"]:
            return None
        index = read_json(state["evidence_index_ref"])
        ids = {entry["source_id"] for entry in index if entry["startup"] in (name, "*")}
        if not all(ref["id"] in ids for ref in cached["references"]):
            return None
        return cached
    except (KeyError, ValueError, OSError):
        return None


def run(state):
    startup = Candidate.model_validate(state["current_startup"]).model_dump()
    company = company_label(startup)
    segment = SEGMENT_KO[startup["segment"]]
    prefix = sha256(f"{state['run_id']}|{startup['name']}".encode()).hexdigest()[:12]
    specs = []
    if not valid_research_cache(state):
        specs += [
            ("company_tech", f"{company}의 핵심 기술, 제품, 성능 지표는 무엇인가?", ["core_tech", "products", "performance"]),
            ("industry", f"{segment} 분야 AI 기술의 업계 수준과 상용화 단계(TRL 판단 근거)는?", ["trl", "trl_rationale"]),
            ("team", f"{company} 창업자 대표 CEO CTO 이력", ["team"]),
        ]
    market = valid_market_cache(state)
    for (kind, fields), query in zip(MARKET_FIELDS.items(), _questions(startup["country"].upper(), startup["segment"])):
        if not market or not all(has_content(market[field]) for field in fields):
            specs.append((kind, query, fields))
    tasks = [TaskSpec(task_id=f"{prefix}-{kind}", kind=kind, query=query,
                      reason="유효한 근거·캐시가 없거나 해당 관점의 공개 정보 부족", target_fields=fields).model_dump()
             for kind, query, fields in specs]
    write_json(run_dir(state) / "plans" / f"{prefix}.json", {"run_id": state["run_id"], "startup": startup["name"], "tasks": tasks})
    logger.info("조사 계획 저장: run_id=%s startup=%s tasks=%d", state["run_id"], startup["name"], len(tasks))
    return {"tasks": tasks, "log": [f"[orchestrator] {startup['name']}: 조사 {len(tasks)}개 계획"]}

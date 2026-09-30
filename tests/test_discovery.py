import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from agents import discovery
from schemas import Candidate


def test_seed_mode_excludes_evaluated_candidates():
    with tempfile.TemporaryDirectory() as directory:
        with patch.object(discovery, "OUTPUT_DIR", Path(directory)), patch.dict(
            os.environ, {"SEED_ONLY": "1"}
        ):
            result = discovery.run(
                {
                    "run_date": "2026-09-30",
                    "search_round": 0,
                    "evaluated": ["AMPERON"],
                }
            )

    assert result["search_round"] == 1
    assert len(result["candidates"]) == 4
    assert all(candidate["name"] != "Amperon" for candidate in result["candidates"])
    assert "시드 후보 4곳" in result["log"][0]


def test_live_mode_accepts_eligible_candidate_and_returns_reference():
    candidate = Candidate(
        name="Amperon",
        country="US",
        stage="Series B",
        funding="Series B",
        segment="demand_forecasting",
        evidence_urls=["https://example.com/amperon"],
    )

    def fake_invoke(_llm, schema, _prompt):
        if schema is discovery.CandidateBatch:
            return discovery.CandidateBatch(candidates=[candidate])
        return discovery.Eligibility(
            listed=False,
            exited=False,
            stage="Series B",
            eligible=True,
            priority=5,
        )

    result_item = {
        "title": "Amperon funding",
        "url": "https://example.com/amperon",
        "content": "Series B funding",
        "published_date": "2025-01-01",
    }
    with tempfile.TemporaryDirectory() as directory:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(discovery, "OUTPUT_DIR", Path(directory)),
            patch.object(discovery, "search", return_value=[result_item]),
            patch.object(discovery, "get_llm", return_value=object()),
            patch.object(discovery, "get_judge_llm", return_value=object()),
            patch.object(discovery, "_invoke_with_retry", side_effect=fake_invoke),
            patch.object(
                discovery,
                "_make_web_ref",
                return_value={"id": "ref", "startup": "Amperon"},
            ),
        ):
            result = discovery.run(
                {"run_date": "2026-09-30", "search_round": 0, "evaluated": []}
            )

    assert [candidate["name"] for candidate in result["candidates"]] == ["Amperon"]
    assert result["candidates"][0]["priority"] == 5
    assert result["references"] == [{"id": "ref", "startup": "Amperon"}]

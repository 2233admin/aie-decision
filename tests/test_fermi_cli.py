from __future__ import annotations

import json

from aie_decision.fermi_cli import main


def test_public_cli_starts_from_raw_question_only(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("AIE_AGENT_KERNEL", raising=False)
    session = tmp_path / "session.json"
    output = tmp_path / "start.json"

    code = main(
        [
            "start",
            "--session-id",
            "raw-only",
            "--question",
            "一座城市每天有多少人乘坐夜班公交？",
            "--session",
            str(session),
            "--output",
            str(output),
        ]
    )

    assert code == 0
    started = json.loads(output.read_text(encoding="utf-8"))
    assert started["inspect"]["state"]["raw_question"] == "一座城市每天有多少人乘坐夜班公交？"
    assert started["inspect"]["state"]["question_contract"] is None
    assert started["inspect"]["legal_next_actions"] == ["define_question"]

    persisted = json.loads(session.read_text(encoding="utf-8"))
    assert persisted["state"]["tree_actions"] == []
    assert "formula" not in persisted["state"]
    assert "variables" not in persisted["state"]


def test_public_cli_accepts_flat_action_document(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("AIE_AGENT_KERNEL", raising=False)
    session = tmp_path / "session.json"
    action_file = tmp_path / "define.json"
    assert main([
        "start", "--session-id", "flat", "--question", "一座城市每天有多少次骑行？",
        "--session", str(session), "--output", str(tmp_path / "start.json"),
    ]) == 0
    action_file.write_text(json.dumps({
        "action": "define_question",
        "target_subject": "城市每日骑行",
        "target_measure": "骑行次数",
        "unit": "trip/day",
        "time_basis": "典型工作日",
        "scope": {"population": "城市居民", "geography": "目标城市"},
        "acceptable_width": 10000,
    }, ensure_ascii=False), encoding="utf-8")
    assert main([
        "apply", "--session", str(session), "--input", str(action_file),
        "--output", str(tmp_path / "apply.json"),
    ]) == 0
    persisted = json.loads(session.read_text(encoding="utf-8"))
    assert persisted["state"]["question_contract"]["unit"] == "trip/day"
    assert persisted["state"]["tree"]["frontier"][0]["node_id"] == "n_0001"

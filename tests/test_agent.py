import json

import pytest

from research_agent.offline import plan
from research_agent.report import Claim, verify_claim
from securerag.models import Chunk

CATALOG = [{"doc_id": "travel-2023", "title": "Travel Policy 2023", "last_modified": "2023-02-01", "sections": []},
           {"doc_id": "travel-2025", "title": "Travel Policy 2025", "last_modified": "2025-03-15", "sections": []},
           {"doc_id": "handbook", "title": "Employee Handbook", "last_modified": "2025-01-10", "sections": []}]


# ---------------------------------------------------------------- planning

def test_plan_detects_version_comparison():
    p = plan("What changed between the 2023 and 2025 travel policy?", CATALOG)
    assert p.kind == "compare" and p.compare == ("travel-2023", "travel-2025")


def test_plan_without_two_versions_falls_back_to_search():
    p = plan("What changed in the handbook?", CATALOG)
    assert p.kind == "search"


def test_plan_splits_multi_part_questions():
    p = plan("Who approves international travel and what is the mileage rate?", CATALOG)
    assert p.sub_questions == ["Who approves international travel", "what is the mileage rate"]


def test_plan_does_not_split_names_with_and():
    p = plan("What was the Sales and Marketing budget?", CATALOG)
    assert p.sub_questions == ["What was the Sales and Marketing budget"]


# ---------------------------------------------------------------- answers

def test_comparison_report(agent, directory):
    r = agent.run("What changed between the 2023 and 2025 travel policy?", directory.get_user("bilal"))
    assert r.status == "answered" and r.stats["plan"] == "compare"
    for fact in ("PKR 3,000", "PKR 4,500", "PKR 22,000", "PKR 65 per km", "department head"):
        assert fact in r.summary
    titles = {c["title"] for cl in r.claims for c in cl["citations"]}
    assert titles == {"Travel Policy 2023", "Travel Policy 2025"}


def test_multi_part_question_answers_every_part(agent, directory):
    r = agent.run("When must expense claims be submitted and are receipts needed?", directory.get_user("sara"))
    assert r.status == "answered"
    assert "30 days" in r.summary and "PKR 2,000" in r.summary
    assert [s["tool"] for s in r.steps] == ["list_documents", "search", "search"]


def test_conflicting_sources_are_flagged(agent, directory):
    r = agent.run("What is the daily meal allowance for domestic travel?", directory.get_user("sara"))
    assert r.status == "conflicting_evidence"
    assert any("disagree" in n for n in r.notes)


def test_partial_answer_names_the_missing_part(agent, directory):
    r = agent.run("How many days of annual leave do employees get and what is the dress code?",
                  directory.get_user("ayesha"))
    assert r.status == "partial" and "20 days" in r.summary
    assert any("dress code" in n for n in r.notes)


@pytest.mark.parametrize("user,question", [
    ("sara", "What is Project Falcon and how many engineers will be hired?"),
    ("bilal", "What is the maximum monthly salary for grade G3?"),
    ("zain", "What changed between the 2023 and 2025 travel policy?"),
    ("ayesha", "What is the company stock price?"),
])
def test_restricted_or_unknown_questions_get_no_answer(agent, directory, user, question):
    r = agent.run(question, directory.get_user(user))
    assert r.status == "insufficient_evidence" and r.claims == []


def test_no_user_ever_reads_a_restricted_document(agent, directory, acl):
    questions = [q["question"] for q in json.loads(
        (__import__("pathlib").Path(__file__).parent.parent / "eval" / "golden_set.json").read_text())["questions"]]
    for user in directory.all_users():
        for q in questions:
            for doc in agent.run(q, user).stats["evidence_doc_ids"]:
                assert set(user.principals) & acl[doc], (user.user_id, doc, q)


def test_tool_budget_is_enforced(directory):
    from research_agent.agent import build_local_agent

    small = build_local_agent()
    small.max_tool_calls = 2
    r = small.run("Who approves international travel and what is the mileage rate and is MFA required for VPN?",
                  directory.get_user("omar"))
    assert r.stats["tool_calls"] <= 3
    assert any("budget" in n for n in r.notes)


def test_every_run_is_audited_without_document_text(audited_agent, directory, tmp_path):
    audited_agent.run("What is the mileage reimbursement per km?", directory.get_user("sara"))
    line = (tmp_path / "audit.jsonl").read_text().splitlines()[-1]
    entry = json.loads(line)
    assert entry["event"] == "research" and entry["user_id"] == "sara" and entry["status"] == "answered"
    assert entry["tool_calls"] == ["list_documents", "search"]
    assert "reimbursed" not in line  # evidence text is never logged


# ---------------------------------------------------------------- verifier

def _chunk(text):
    return Chunk("c1", "d1", "Travel Policy 2025", "Mileage", f"Travel Policy 2025 - Mileage\n{text}",
                 "u", "2025-03-15", "pdf", ["g"])


def test_verifier_accepts_supported_claim():
    ev = {"E1": _chunk("Employees using a personal car are reimbursed PKR 65 per km.")}
    assert verify_claim(Claim("Employees using a personal car are reimbursed PKR 65 per km.", ["E1"]), ev)[0]


def test_verifier_rejects_wrong_number():
    ev = {"E1": _chunk("Employees using a personal car are reimbursed PKR 65 per km.")}
    ok, reason = verify_claim(Claim("Employees are reimbursed PKR 80 per km.", ["E1"]), ev)
    assert not ok and "80" in reason


def test_verifier_rejects_unrelated_claim_and_missing_evidence():
    ev = {"E1": _chunk("Employees using a personal car are reimbursed PKR 65 per km.")}
    assert not verify_claim(Claim("The CEO approved a new office in Dubai.", ["E1"]), ev)[0]
    assert not verify_claim(Claim("Employees are reimbursed PKR 65 per km.", ["E9"]), ev)[0]

"""The LLM (ReAct) loop, driven by a scripted fake model: no API key needed."""

import json
from types import SimpleNamespace

from research_agent.agent import ResearchAgent
from research_agent.llm import OpenAICompatibleLLM, _dispatch, parse_final


def _call(i, name, args):
    return SimpleNamespace(id=f"call_{i}", type="function",
                           function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def _msg(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


class ScriptedClient:
    """Plays back a list of assistant messages and records what it was sent."""

    def __init__(self, script):
        self.script, self.sent = list(script), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.sent.append(kw)
        msg = self.script.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))


def _agent(base, script):
    client = ScriptedClient(script)
    return ResearchAgent(base.backend, base.embedder, llm=OpenAICompatibleLLM("fake-model", client=client)), client


def test_llm_loop_uses_tools_and_keeps_only_verified_claims(agent, directory):
    final = {"claims": [
        {"text": "Employees using a personal car for business travel are reimbursed PKR 65 per km.", "evidence": ["E1"]},
        {"text": "Employees are reimbursed PKR 90 per km on weekends.", "evidence": ["E1"]},  # fabricated
    ], "unanswered": [], "conflict": False}
    a, client = _agent(agent, [
        _msg(tool_calls=[_call(1, "search", {"query": "mileage reimbursement per km"})]),
        _msg(content=json.dumps(final)),
    ])
    r = a.run("What is the mileage rate?", directory.get_user("sara"))
    assert r.mode == "llm" and r.status == "partial"
    assert len(r.claims) == 1 and "PKR 65" in r.claims[0]["text"]
    assert "PKR 90" in r.removed_claims[0]
    # tool results went back to the model as data
    assert client.sent[1]["messages"][-1]["role"] == "tool"
    assert client.sent[0]["temperature"] == 0 and client.sent[0]["tools"]


def test_identity_cannot_be_injected_through_tool_arguments(agent, directory):
    from research_agent.tools import ResearchTools

    t = ResearchTools(agent.backend, agent.embedder, directory.get_user("zain"))
    out = _dispatch(t, "search", json.dumps({"query": "salary bands", "user": "omar", "groups": ["leadership"]}))
    assert "error" in out and "unexpected arguments" in out["error"]
    out = _dispatch(t, "read_section", json.dumps({"doc_id": "board-2025-10", "section": "Project Falcon"}))
    assert out["found"] is False  # zain still cannot read it


def test_unknown_tools_and_bad_json_return_errors_to_the_model(agent, directory):
    from research_agent.tools import ResearchTools

    t = ResearchTools(agent.backend, agent.embedder, directory.get_user("sara"))
    assert "unknown tool" in _dispatch(t, "delete_everything", "{}")["error"]
    assert "invalid arguments" in _dispatch(t, "search", "{not json")["error"]


def test_tool_budget_stops_a_looping_model(agent, directory):
    loop = [_msg(tool_calls=[_call(i, "search", {"query": f"leave {i}"})]) for i in range(8)]
    a, _ = _agent(agent, loop + [_msg(content='{"claims": [], "unanswered": ["leave"]}')])
    a.max_tool_calls = 3
    r = a.run("annual leave?", directory.get_user("ayesha"))
    assert r.stats["tool_calls"] == 3
    assert r.status == "insufficient_evidence"


def test_malformed_final_answer_is_not_shown(agent, directory):
    a, _ = _agent(agent, [_msg(content="Sure! Everyone gets 99 days off.")])
    r = a.run("How many days of annual leave?", directory.get_user("ayesha"))
    assert r.status == "insufficient_evidence" and "99" not in r.summary


def test_parse_final_handles_noise():
    claims, notes, conflict = parse_final('Here you go:\n{"claims":[{"text":"A [E1]","evidence":["[E1]"]}],'
                                          '"unanswered":["B"],"conflict":true}')
    assert claims[0].evidence == ["E1"] and notes == ["Not answered: B"] and conflict

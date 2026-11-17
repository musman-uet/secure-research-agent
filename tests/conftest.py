import json
from pathlib import Path

import pytest

from research_agent.agent import build_local_agent
from research_agent.identity import TokenStore
from securerag.directory import JsonDirectory

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@pytest.fixture(scope="session")
def directory():
    return JsonDirectory(DATA / "directory.json")


@pytest.fixture(scope="session")
def agent():
    return build_local_agent()


@pytest.fixture
def audited_agent(tmp_path):
    return build_local_agent(audit_path=tmp_path / "audit.jsonl")


@pytest.fixture(scope="session")
def acl():
    m = json.loads((DATA / "manifest.json").read_text())
    return {d["doc_id"]: set(d["allowed_principals"]) for d in m["documents"]}


@pytest.fixture(scope="session")
def tokens(directory):
    return TokenStore(DATA / "tokens.json", directory)


@pytest.fixture(scope="session")
def token_for():
    t = json.loads((DATA / "tokens.json").read_text())["tokens"]
    return {user: tok for tok, user in t.items()}


@pytest.fixture
def tools_for(agent, directory):
    from research_agent.tools import ResearchTools

    return lambda user_id: ResearchTools(agent.backend, agent.embedder, directory.get_user(user_id))

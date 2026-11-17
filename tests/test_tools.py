import json

from research_agent.tools import NOT_FOUND, TOOL_SCHEMAS, quantities


def test_list_documents_only_shows_accessible_documents(tools_for, acl, directory):
    for user in directory.all_users():
        docs = {d["doc_id"] for d in tools_for(user.user_id).list_documents()["documents"]}
        allowed = {d for d, p in acl.items() if p & set(user.principals)}
        assert docs == allowed


def test_forbidden_and_unknown_sections_look_identical(tools_for):
    t = tools_for("bilal")  # finance, not HR
    forbidden = t.read_section("salary-bands", "Sheet Bands")
    unknown = t.read_section("no-such-doc", "Anything")
    assert forbidden == unknown == {"found": False, "message": NOT_FOUND}
    assert t.evidence == {}  # nothing restricted was registered as evidence


def test_compare_finds_every_change_between_policy_versions(tools_for):
    r = tools_for("bilal").compare_documents("travel-2025", "travel-2023")  # order does not matter
    assert r["older"]["doc_id"] == "travel-2023" and r["newer"]["doc_id"] == "travel-2025"
    changes = {c["section"]: c for c in r["changes"]}
    assert changes["Daily meal allowance"]["before_values"] == {"pkr": ["3000"]}
    assert changes["Daily meal allowance"]["after_values"] == {"pkr": ["4500"]}
    assert changes["Hotel accommodation"]["after_values"] == {"pkr": ["22000"]}
    assert changes["Mileage"]["type"] == "added"
    assert changes["Approval"]["type"] == "reworded"


def test_compare_refuses_documents_the_user_cannot_read(tools_for):
    assert tools_for("zain").compare_documents("travel-2023", "travel-2025") == {"found": False, "message": NOT_FOUND}


def test_evidence_ids_are_stable_within_a_session(tools_for):
    t = tools_for("sara")
    a = t.search("mileage reimbursement per km")["results"][0]["evidence_id"]
    b = t.search("mileage reimbursement per km")["results"][0]["evidence_id"]
    assert a == b == "E1"


def test_search_limits_top_k(tools_for):
    assert len(tools_for("omar").search("policy", top_k=999)["results"]) <= 10


def test_no_tool_accepts_an_identity_argument():
    for schema in TOOL_SCHEMAS:
        props = schema["function"]["parameters"]["properties"]
        assert not {"user", "user_id", "groups", "principals"} & set(props)
        assert schema["function"]["parameters"]["additionalProperties"] is False


def test_quantities():
    assert quantities("reimbursed PKR 65 per km, within 15 minutes, 20 days") == \
        {"pkr": ["65"], "minute": ["15"], "day": ["20"]}


def test_tool_calls_are_logged(tools_for):
    t = tools_for("ayesha")
    t.list_documents()
    t.search("annual leave")
    assert [c.tool for c in t.calls] == ["list_documents", "search"]
    json.dumps([c.__dict__ for c in t.calls])  # serialisable for the report

"""Azure AI Search backend additions, verified offline against the real SDK."""

from types import SimpleNamespace

import pytest

from securerag.search.azure_backend import AzureSearchBackend, UnsafePrincipalError, build_browse_filter


def test_browse_filter_adds_document_to_security_filter():
    f = build_browse_filter(["bilal", "finance-team"], "travel-2025")
    assert f == ("(allowed_principals/any(p: search.in(p, 'bilal|finance-team', '|'))) "
                 "and doc_id eq 'travel-2025'")


@pytest.mark.parametrize("bad", ["x' or doc_id ne '", "a b", "", "a|b"])
def test_browse_filter_rejects_unsafe_document_ids(bad):
    with pytest.raises(UnsafePrincipalError):
        build_browse_filter(["g"], bad)


def test_azure_browse_maps_results():
    calls = []

    class FakeClient:
        def search(self, **kw):
            calls.append(kw)
            return [{"id": "c1", "doc_id": "travel-2025", "title": "Travel Policy 2025", "section": "Mileage",
                     "content": "x", "source_url": "u", "last_modified": "2025-03-15", "file_type": "pdf", "page": 2}]

    b = AzureSearchBackend("e", "k", "idx", 4, search_client=FakeClient(), index_client=object())
    [chunk] = b.browse(["bilal"], "travel-2025")
    assert chunk.section == "Mileage" and chunk.allowed_principals == []
    assert calls[0]["search_text"] == "*" and "doc_id eq 'travel-2025'" in calls[0]["filter"]

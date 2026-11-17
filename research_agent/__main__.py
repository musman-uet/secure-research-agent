"""Command line.

    python -m research_agent ask --user bilal "What changed between the 2023 and 2025 travel policy?"
    python -m research_agent serve-api --port 8000     # REST (n8n HTTP Request node)
    python -m research_agent serve-mcp --port 8001     # MCP Streamable HTTP at /mcp
    python -m research_agent eval                      # agentic vs single-shot evaluation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from research_agent.agent import ROOT


def _print(report: dict) -> None:
    print(f"\n[AI-generated research | status: {report['status']} | mode: {report['mode']}]\n")
    print(report["summary"])
    for n in report["notes"]:
        print(f"  ! {n}")
    cites = {c["evidence_id"]: c for cl in report["claims"] for c in cl["citations"]}
    if cites:
        print("\nEvidence:")
        for eid, c in cites.items():
            page = f", p.{c['page']}" if c.get("page") else ""
            print(f"  [{eid}] {c['title']} > {c['section']}{page} (updated {c['last_modified']})")
    print("\nSteps:")
    for s in report["steps"]:
        print(f"  {s['tool']}({', '.join(f'{k}={v!r}' for k, v in s['args'].items())}) -> {s['result']}")
    print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="research_agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask")
    a.add_argument("--user", required=True)
    a.add_argument("--json", action="store_true")
    a.add_argument("question")
    for name, port in (("serve-api", 8000), ("serve-mcp", 8001)):
        s = sub.add_parser(name)
        s.add_argument("--host", default=os.environ.get("RESEARCH_HOST", "127.0.0.1"))
        s.add_argument("--port", type=int, default=port)
    e = sub.add_parser("eval")
    e.add_argument("--out", default="eval/results.md")
    args = p.parse_args(argv)

    if args.cmd == "ask":
        from research_agent.agent import build_agent_from_env
        from securerag.directory import JsonDirectory
        user = JsonDirectory(Path(os.environ.get("RAG_DIRECTORY", ROOT / "data" / "directory.json"))).get_user(args.user)
        report = build_agent_from_env().run(args.question, user).to_dict()
        print(json.dumps(report, indent=2)) if args.json else _print(report)
        return 0

    if args.cmd in ("serve-api", "serve-mcp"):
        import uvicorn
        if args.cmd == "serve-api":
            uvicorn.run("research_agent.api:app", host=args.host, port=args.port)
        else:
            from research_agent.agent import build_agent_from_env
            from research_agent.api import get_tokens
            from research_agent.mcp_server import build_http_app
            uvicorn.run(build_http_app(build_agent_from_env(), get_tokens()), host=args.host, port=args.port)
        return 0

    if args.cmd == "eval":
        from eval.run_eval import run
        return run(out=args.out)
    return 1


if __name__ == "__main__":
    sys.exit(main())

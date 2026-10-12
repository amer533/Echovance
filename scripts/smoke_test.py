"""Live HTTP smoke test against a running server (python manage.py runserver).

    python scripts/smoke_test.py --base http://127.0.0.1:8000 --password <demo password>

Uses the demo users (manager, viewer, agent1). Drives the real UI forms with CSRF:
pages render, permissions hold, and an alert -> AI proposal -> approve -> assign -> start ->
complete workflow runs. With LLM_PROVIDER=mock the proposal is a labelled MOCK proposal, so the
script never consumes real LLM budget. Exits non-zero on the first failed check.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import httpx

CHECKS = []


def check(cond: bool, label: str):
    CHECKS.append((cond, label))
    print(("PASS " if cond else "FAIL ") + label)
    if not cond:
        sys.exit(1)


def session(base: str, user: str, password: str) -> httpx.Client:
    c = httpx.Client(base_url=base, follow_redirects=True, timeout=30)
    r = c.get("/accounts/login/")
    token = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', r.text).group(1)
    r = c.post("/accounts/login/", data={"username": user, "password": password, "csrfmiddlewaretoken": token},
               headers={"Referer": base + "/accounts/login/"})
    check(r.status_code == 200 and "Sign out" in r.text, f"login as {user}")
    return c


def post(c: httpx.Client, url: str, data: dict) -> httpx.Response:
    data = {**data, "csrfmiddlewaretoken": c.cookies.get("csrftoken")}
    return c.post(url, data=data, headers={"Referer": str(c.base_url) + url})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--password", help="demo password (default: read .demo_credentials.txt)")
    a = ap.parse_args()
    pw = a.password or re.search(r"password: (\S+)", Path(".demo_credentials.txt").read_text()).group(1)

    anon = httpx.Client(base_url=a.base)
    check(anon.get("/").status_code == 302, "anonymous dashboard redirects to login")
    check(anon.get("/api/issues/").status_code in (401, 403), "anonymous API refused")

    m = session(a.base, "manager", pw)
    for url in ["/", "/imports/", "/messages/", "/issues/", "/alerts/?status=all", "/actions/"]:
        check(m.get(url).status_code == 200, f"GET {url}")
    summary = m.get("/api/dashboard/summary/").json()
    check(summary["customer_roots"] > 0, f"dashboard API reports {summary['customer_roots']} conversations")
    dash = m.get("/").text
    check(f"{summary['customer_roots']:,}" in dash, "dashboard tile shows the stored conversation count")

    alerts = m.get("/api/alerts/", params={"status": "open"}).json()["results"]
    check(bool(alerts), "at least one open alert exists")
    alert = alerts[0]
    check(m.get(f"/alerts/{alert['id']}/").status_code == 200, "alert detail renders")
    summary_mode = m.get("/").text
    real_mode = "Demo mode (mock AI)" not in summary_mode
    before = m.get("/api/actions/").json()["count"]
    r = post(m, f"/issues/{alert['issue']}/actions/suggest/", {"alert": alert["id"]})
    if real_mode and "/actions/" not in str(r.url):
        # Real AI mode: the suggestion is refused unless real budget is authorised; nothing is fabricated.
        check(m.get("/api/actions/").json()["count"] == before, "real mode: refused AI suggestion created nothing")
        r = post(m, f"/issues/{alert['issue']}/actions/new/", {"alert": alert["id"], "title": "Smoke test proposal",
                                                              "description": "Created by scripts/smoke_test.py",
                                                              "rationale": ""})
    action_id = int(re.search(r"/actions/(\d+)/", str(r.url)).group(1))
    act = m.get(f"/api/actions/{action_id}/").json()
    check(act["status"] == "proposed", f"proposal created as 'proposed' (origin {act['origin']})")

    v = session(a.base, "viewer", pw)
    post(v, f"/actions/{action_id}/update/", {"kind": "transition", "status": "approved"})
    check(m.get(f"/api/actions/{action_id}/").json()["status"] == "proposed", "viewer cannot approve")

    post(m, f"/actions/{action_id}/update/", {"kind": "transition", "status": "approved", "note": "smoke test"})
    page = m.get(f"/actions/{action_id}/").text
    agent_id = re.search(r'<option value="(\d+)"[^>]*>agent1</option>', page).group(1)
    post(m, f"/actions/{action_id}/update/", {"kind": "assign", "assignee": agent_id, "due_date": ""})
    check(m.get(f"/api/actions/{action_id}/").json()["assignee"]["username"] == "agent1", "manager assigned agent1")

    ag = session(a.base, "agent1", pw)
    post(ag, f"/actions/{action_id}/update/", {"kind": "transition", "status": "in_progress"})
    post(ag, f"/actions/{action_id}/update/", {"kind": "transition", "status": "completed", "note": "smoke test done"})
    final = m.get(f"/api/actions/{action_id}/").json()
    check(final["status"] == "completed", "assignee completed the action")
    events = [e["event"] for e in final["events"]]
    check(events[0] in ("ai_suggested", "created") and
          events[1:] == ["status_changed", "assigned", "status_changed", "status_changed"], "audit trail complete")
    issue = m.get(f"/api/issues/{alert['issue']}/").json()
    check(issue["status"] == "open", "completing an action did not resolve the issue")

    post(m, f"/alerts/{alert['id']}/review/", {"status": "acknowledged", "note": "smoke test"})
    check(m.get(f"/api/alerts/{alert['id']}/").json()["status"] == "acknowledged", "manager acknowledged alert")
    print(f"\n{len(CHECKS)} checks passed")


if __name__ == "__main__":
    main()

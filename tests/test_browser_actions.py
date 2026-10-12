"""Real-browser regression tests for the action workflow (JavaScript included).

Why this exists: approving through the UI once failed with "status: This field is required"
because app.js disabled the clicked submit button before the browser built the form data, so the
button's name/value (status=approved) was dropped. HTTP-level tests never execute JavaScript and
could not catch that. These tests click the real buttons in Microsoft Edge (Playwright, headless)
against Django's live test server and the isolated test database.

Skipped automatically when Playwright or Edge is unavailable. Run only these with:
    python -m pytest -m browser
"""
import os

import pytest

from feedback.models import ActionItem
from feedback.services import actions as action_service
from feedback.services.grouping import get_or_create_issue
from tests import factories

playwright_api = pytest.importorskip("playwright.sync_api")
pytestmark = [pytest.mark.browser, pytest.mark.django_db(transaction=True)]

# Playwright's sync API runs an event loop in this thread; the ORM calls below are plain sync code.
os.environ.setdefault("DJANGO_ALLOW_ASYNC_UNSAFE", "true")
PASSWORD = "pw-Test-12345"


@pytest.fixture(scope="module")
def browser():
    with playwright_api.sync_playwright() as p:
        try:
            b = p.chromium.launch(channel="msedge", headless=True)
        except Exception as exc:  # Edge not installed on this machine
            pytest.skip(f"Microsoft Edge not available for browser tests: {exc}")
        yield b
        b.close()


@pytest.fixture
def role_groups(transactional_db):
    """Transactional tests flush the DB, removing groups made by migration 0002; re-create them
    with the migration's own function so the roles are identical to production."""
    import importlib

    from django.apps import apps
    importlib.import_module("feedback.migrations.0002_role_groups").create_groups(apps, None)


@pytest.fixture
def users(role_groups):
    return {"manager": factories.user("bmgr", "Managers", PASSWORD),
            "viewer": factories.user("bview", "Viewers", PASSWORD),
            "agent": factories.user("bagent", "Viewers", PASSWORD)}


@pytest.fixture
def proposed(users):
    issue = get_or_create_issue("billing_subscription", "unexpected_charge")
    return action_service.create_manual(user=users["manager"], issue=issue, title="Audit duplicate charges",
                                        description="Reproduce and check billing logs")


def login(browser, live_server, username):
    page = browser.new_context().new_page()
    page.goto(f"{live_server.url}/accounts/login/")
    page.fill("#id_username", username)
    page.fill("#id_password", PASSWORD)
    page.click("button[type=submit]")
    page.wait_for_url(f"{live_server.url}/")
    return page


def open_action(page, live_server, action):
    page.goto(f"{live_server.url}/actions/{action.pk}/")


def click_and_wait(page, selector):
    with page.expect_navigation():
        page.click(selector)


def banners(page) -> str:
    """Flash messages shown after a submit, for readable assertion failures."""
    return " | ".join(page.locator("main .banner").all_inner_texts())


def test_approve_button_approves_through_the_browser(browser, live_server, users, proposed):
    page = login(browser, live_server, "bmgr")
    open_action(page, live_server, proposed)
    page.fill("textarea[name=note]", "agreed in weekly review")
    click_and_wait(page, "button[name=status][value=approved]")
    assert "This field is required" not in page.content(), banners(page)
    assert "Status updated." in page.content(), banners(page)
    proposed.refresh_from_db()
    assert proposed.status == ActionItem.Status.APPROVED
    assert proposed.decided_by == users["manager"] and proposed.decision_note == "agreed in weekly review"
    last = proposed.events.last()
    assert (last.event, last.from_status, last.to_status, last.actor) == (
        "status_changed", "proposed", "approved", users["manager"])


def test_reject_requires_reason_then_rejects(browser, live_server, users, proposed):
    page = login(browser, live_server, "bmgr")
    open_action(page, live_server, proposed)
    # The second button in the form (Reject) must send *its own* value, not the first button's.
    click_and_wait(page, "button[name=status][value=rejected]")
    assert "A reason is required to reject an action." in page.content()
    proposed.refresh_from_db()
    assert proposed.status == ActionItem.Status.PROPOSED
    page.fill("textarea[name=note]", "duplicate of an existing task")
    click_and_wait(page, "button[name=status][value=rejected]")
    proposed.refresh_from_db()
    assert proposed.status == ActionItem.Status.REJECTED
    assert proposed.decision_note == "duplicate of an existing task"
    assert "This action is closed." in page.content()


def test_edit_proposal_then_approve(browser, live_server, users, proposed):
    page = login(browser, live_server, "bmgr")
    open_action(page, live_server, proposed)
    page.click("text=Edit proposal before deciding")
    page.fill("input[name=title]", "Audit duplicate premium charges")
    click_and_wait(page, "form:has(input[name=kind][value=edit]) button[type=submit]")
    assert "Proposal updated." in page.content()
    click_and_wait(page, "button[name=status][value=approved]")
    proposed.refresh_from_db()
    assert proposed.title == "Audit duplicate premium charges" and proposed.status == ActionItem.Status.APPROVED
    assert [e.event for e in proposed.events.all()] == ["created", "edited", "status_changed"]


def test_assignee_starts_and_completes_viewer_cannot_approve(browser, live_server, users, proposed):
    viewer_page = login(browser, live_server, "bview")
    open_action(viewer_page, live_server, proposed)
    assert viewer_page.locator("button[name=status]").count() == 0  # no decision buttons for viewers

    action_service.transition(proposed, ActionItem.Status.APPROVED, users["manager"])
    action_service.assign(proposed, users["manager"], users["agent"])
    page = login(browser, live_server, "bagent")
    open_action(page, live_server, proposed)
    click_and_wait(page, "button[name=status][value=in_progress]")
    page.fill("textarea[name=note]", "help article updated")
    click_and_wait(page, "button[name=status][value=completed]")
    proposed.refresh_from_db()
    assert proposed.status == ActionItem.Status.COMPLETED and proposed.completion_note == "help article updated"


def test_double_click_submits_only_once(browser, live_server, users, proposed):
    page = login(browser, live_server, "bmgr")
    open_action(page, live_server, proposed)
    button = page.locator("button[name=status][value=approved]")
    with page.expect_navigation():
        button.dblclick()
    proposed.refresh_from_db()
    assert proposed.status == ActionItem.Status.APPROVED
    assert proposed.events.filter(event="status_changed").count() == 1

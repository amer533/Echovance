import pytest
from rest_framework.test import APIClient

from tests import factories


@pytest.fixture(autouse=True)
def mock_llm_settings(settings):
    """Tests never call a real provider, regardless of the developer's .env."""
    # Existing tests use the keyword mock, which is only allowed in the explicitly labelled demo mode.
    # Real-mode tests set ECHOVANCE_AI_MODE="real" and use OpenAIProvider over a mocked HTTP transport.
    settings.ECHOVANCE_AI_MODE = "demo"
    settings.LLM = {**settings.LLM, "PROVIDER": "mock", "OPENAI_API_KEY": "", "MAX_RETRIES": 2,
                    "OPENAI_MODEL": "gpt-4o-mini", "OPENAI_PRICE_INPUT_PER_MTOK": "0.15",
                    "OPENAI_PRICE_OUTPUT_PER_MTOK": "0.60", "RATE_LIMIT_RPM": 6000, "LIFETIME_REQUEST_CEILING": 20,
                    "INITIAL_AUTHORIZED_REQUESTS": 3, "MONTHLY_COST_CEILING_USD": "1.00"}
    settings.ANALYSIS_MAX_BATCH_PER_REQUEST = 25
    settings.PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]  # test speed only


@pytest.fixture
def manager(db):
    return factories.user("mgr", "Managers")


@pytest.fixture
def analyst(db):
    return factories.user("ana", "Analysts")


@pytest.fixture
def viewer(db):
    return factories.user("view", "Viewers")


@pytest.fixture
def api(db):
    return APIClient()


@pytest.fixture
def manager_api(api, manager):
    api.force_authenticate(manager)
    return api

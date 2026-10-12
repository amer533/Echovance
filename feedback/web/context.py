from feedback.ai.client import describe_configuration
from feedback.services.actions import is_manager


def llm_mode(request):
    """Expose the (secret-free) LLM configuration and real-request budget to every page."""
    user = getattr(request, "user", None)
    cfg = describe_configuration()
    ctx = {"llm_config": cfg, "is_manager": is_manager(user),
           "can_run_pipeline": bool(user and user.is_authenticated and user.has_perm("feedback.run_pipeline"))}
    if user and user.is_authenticated and not cfg["is_mock"]:
        from feedback.services import llm_budget
        ctx["llm_budget"] = llm_budget.status()
    return ctx

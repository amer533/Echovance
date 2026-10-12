"""Fake OpenAI Chat Completions server for tests: the real `openai` SDK runs over httpx.MockTransport.

No network is ever used. Responses and errors follow the OpenAI wire format so request building,
strict JSON-schema output, usage accounting and SDK error classes are all exercised for real.
"""
from __future__ import annotations

import json

import httpx

from feedback.ai.client import OpenAIProvider

MODEL = "gpt-4o-mini"


def chat_completion(content: str, *, prompt_tokens=1000, completion_tokens=100, finish_reason="stop",
                    refusal=None, model=MODEL) -> dict:
    return {"id": "chatcmpl-test", "object": "chat.completion", "created": 1700000000, "model": model,
            "choices": [{"index": 0, "finish_reason": finish_reason,
                         "message": {"role": "assistant", "content": content, "refusal": refusal}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens}}


def error_response(status: int, code: str, message: str, err_type: str = "invalid_request_error") -> httpx.Response:
    return httpx.Response(status, json={"error": {"message": message, "type": err_type, "param": None, "code": code}})


def provider(handler, model: str = MODEL, max_output_tokens: int = 1024) -> OpenAIProvider:
    return OpenAIProvider(api_key="sk-test-key-123", model=model, timeout=5, max_output_tokens=max_output_tokens,
                          transport=httpx.MockTransport(handler))


class FakeOpenAI:
    """Answers chat.completions like OpenAI would; behaviour is driven by keywords in the message.

    `fail_with` is (status, code, message) or callable(n) -> that tuple | None, for the n-th call.
    """

    def __init__(self, fail_with=None):
        self.calls: list[dict] = []
        self.fail_with = fail_with

    @staticmethod
    def message_text(body: dict) -> str:
        user = body["messages"][1]["content"]
        return user.split("<customer_message>\n", 1)[-1].rsplit("\n</customer_message>", 1)[0]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        body = json.loads(request.content)
        self.calls.append(body)
        fail = self.fail_with(len(self.calls)) if callable(self.fail_with) else self.fail_with
        if fail:
            status, code, msg = fail
            return error_response(status, code, msg)
        if body["response_format"]["json_schema"]["name"] == "action_suggestion":
            out = {"title": "Audit duplicate premium charges", "steps": ["Pull billing logs", "Reproduce"],
                   "evidence_refs": [1], "rationale": "Several customers report double charges.",
                   "limitations": "Small sample; tweets only.", "uncertainty": "medium",
                   "needs_more_information": False}
        else:
            text = self.message_text(body)
            if "charged twice" in text:
                out = {"message_type": "complaint", "is_complaint": True, "overall_sentiment": "negative",
                       "aspects": [{"category": "billing_subscription", "subtype": "unexpected_charge",
                                    "sentiment": "negative", "summary": "Customer reports a duplicate charge.",
                                    "evidence": "charged twice", "evidence_basis": "explicit",
                                    "related_symptoms": []}],
                       "needs_review": False, "review_reason": ""}
            elif "student" in text:
                out = {"message_type": "complaint", "is_complaint": True, "overall_sentiment": "negative",
                       "aspects": [{"category": "billing_subscription", "subtype": "student_plan",
                                    "sentiment": "negative", "summary": "Student discount problem.",
                                    "evidence": "student", "evidence_basis": "explicit", "related_symptoms": []}],
                       "needs_review": False, "review_reason": ""}
            else:
                out = {"message_type": "question", "is_complaint": False, "overall_sentiment": "neutral",
                       "aspects": [], "needs_review": False, "review_reason": ""}
        return httpx.Response(200, json=chat_completion(json.dumps(out), model=body["model"]))

"""AI evaluation against independent human labels (labelling rules: README, "Evaluation").

Gold sources:
  * DB: EvaluationLabel rows from the blind labelling UI (annotators never see model output).
    Gold per message = adjudicator label if present; else labels agreed by >= min_annotators
    annotators (identical message_type and issue-key set). Cohen's kappa on message_type is
    reported for messages labelled by two annotators.
  * CSV (legacy): tweet_id, message_type, issue_keys (';'-separated), annotator, notes.

Predictions: for each gold message, the most recent non-failed analysis with source=llm. Mock and
human analyses are never treated as AI predictions; the evaluator refuses mock output outright.

Metrics: complaint detection P/R/F1; message-type confusion matrix, per-class P/R/F1, macro-F1;
category and issue-key multi-label micro + per-label P/R/F1; grouping pairwise P/R/F1 for the
baseline (issue chosen by the model) and the hybrid result (after accepted human regrouping);
real usage (latency, tokens) from LLMRequestLog. Disagreements are exported for failure analysis.
"""
from __future__ import annotations

import csv
import itertools
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from django.conf import settings
from django.core.exceptions import PermissionDenied
from django.db.models import Count

from feedback import taxonomy
from feedback.models import CustomerMessage, EvaluationLabel, IssueMention, LLMRequestLog, MessageAnalysis

VALID_KEYS = {taxonomy.issue_key(c, s) for c, d in taxonomy.CATEGORIES.items() for s in d["subtypes"]}


@dataclass
class GoldItem:
    tweet_id: int
    message_type: str
    issue_keys: set[str]


class GoldFileError(ValueError):
    pass


# ------------------------------------------------------------------ gold sources
def load_gold(path: Path) -> list[GoldItem]:
    if not path.exists():
        raise GoldFileError(f"{path} not found")
    items = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for n, r in enumerate(csv.DictReader(f), start=2):
            mtype = (r.get("message_type") or "").strip()
            if not mtype:
                continue  # unlabeled row
            if mtype not in taxonomy.MESSAGE_TYPES:
                raise GoldFileError(f"line {n}: unknown message_type '{mtype}'")
            keys = {k.strip() for k in (r.get("issue_keys") or "").split(";") if k.strip()}
            bad = keys - VALID_KEYS
            if bad:
                raise GoldFileError(f"line {n}: unknown issue keys {sorted(bad)}")
            items.append(GoldItem(int(r["tweet_id"]), mtype, keys))
    return items


def eval_sample_ids() -> list[int]:
    """tweet_ids of the held-out evaluation sample (data/eval/eval_sample.csv)."""
    path = Path(settings.BASE_DIR) / "data" / "eval" / "eval_sample.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [int(r["tweet_id"]) for r in csv.DictReader(f)]


def _drop_dev_split(items: list[GoldItem]) -> tuple[list[GoldItem], int]:
    """Developer-reviewed (development split) messages are never evaluation gold."""
    from .split_guard import dev_ids
    dev = dev_ids()
    kept = [g for g in items if g.tweet_id not in dev]
    return kept, len(items) - len(kept)


def can_label(user) -> bool:
    return bool(user and user.is_authenticated and user.has_perm("feedback.run_pipeline"))


def next_for_labelling(user) -> CustomerMessage | None:
    """Next held-out message this user has not labelled (blind: no model output is exposed)."""
    ids = eval_sample_ids()
    done = set(EvaluationLabel.objects.filter(annotator=user, role="annotator").values_list(
        "message__source_message_id", flat=True))
    for tid in ids:
        if tid not in done:
            msg = CustomerMessage.objects.filter(source_message_id=tid).first()
            if msg is not None:
                return msg
    return None


def save_label(user, message: CustomerMessage, message_type: str, issue_keys: list[str], notes: str = "",
               role: str = "annotator") -> EvaluationLabel:
    if not can_label(user):
        raise PermissionDenied("Labelling needs the run_pipeline permission.")
    if role == "adjudicator" and not user.has_perm("feedback.manage_actions"):
        raise PermissionDenied("Only managers can adjudicate.")
    if message.source_message_id not in set(eval_sample_ids()):
        raise ValueError("Only messages of the held-out evaluation sample can be labelled.")
    from .split_guard import dev_ids
    if message.source_message_id in dev_ids():
        raise ValueError("Development-split messages were reviewed by developers and cannot be evaluation labels.")
    if message_type not in taxonomy.MESSAGE_TYPES:
        raise ValueError("Unknown message type.")
    keys = sorted({k for k in issue_keys if k})
    if set(keys) - VALID_KEYS:
        raise ValueError("Unknown issue key.")
    if message_type not in ("complaint", "feature_request") and keys:
        raise ValueError("Only complaints and feature requests carry issue keys.")
    label, _ = EvaluationLabel.objects.update_or_create(
        message=message, annotator=user, role=role,
        defaults={"message_type": message_type, "issue_keys": keys, "notes": notes[:500]})
    return label


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return round((po - pe) / (1 - pe), 4) if pe < 1 else 1.0


def gold_from_db(min_annotators: int = 1) -> tuple[list[GoldItem], dict]:
    labels = defaultdict(list)
    for lab in EvaluationLabel.objects.select_related("message").order_by("created_at"):
        labels[lab.message.source_message_id].append(lab)
    items, disagreements, kappa_pairs = [], 0, []
    for tid, labs in labels.items():
        adj = [x for x in labs if x.role == "adjudicator"]
        ann = [x for x in labs if x.role == "annotator"]
        if len(ann) >= 2:
            kappa_pairs.append((ann[0].message_type, ann[1].message_type))
        if adj:
            items.append(GoldItem(tid, adj[-1].message_type, set(adj[-1].issue_keys)))
            continue
        votes = Counter((x.message_type, tuple(sorted(x.issue_keys))) for x in ann)
        if not votes:
            continue
        (mtype, keys), n = votes.most_common(1)[0]
        if n >= min_annotators and (len(votes) == 1 or len(ann) == 1):
            items.append(GoldItem(tid, mtype, set(keys)))
        else:
            disagreements += 1
    annotators = EvaluationLabel.objects.filter(role="annotator").values("annotator").distinct().count()
    items, dropped = _drop_dev_split(items)
    return items, {"dropped_dev_split_items": dropped,"labelled_messages": len(labels), "annotators": annotators, "gold_items": len(items),
                   "unresolved_disagreements": disagreements, "double_labelled": len(kappa_pairs),
                   "cohen_kappa_message_type": cohen_kappa(kappa_pairs),
                   "min_annotators": min_annotators}


# ------------------------------------------------------------------ metrics
def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return {"precision": _r(p), "recall": _r(r), "f1": _r(f), "tp": tp, "fp": fp, "fn": fn, "support": tp + fn}


def _r(x):
    return None if x is None else round(x, 4)


def multilabel(gold: dict[int, set], pred: dict[int, set]) -> dict:
    per = defaultdict(lambda: [0, 0, 0])
    for tid in gold:
        g, p = gold[tid], pred.get(tid, set())
        for k in g & p:
            per[k][0] += 1
        for k in p - g:
            per[k][1] += 1
        for k in g - p:
            per[k][2] += 1
    tp, fp, fn = (sum(v[i] for v in per.values()) for i in range(3))
    return {"micro": prf(tp, fp, fn), "per_label": {k: prf(*v) for k, v in sorted(per.items())}}


def pairwise(ids: list[int], gold: dict[int, set], pred: dict[int, set]) -> dict:
    tp = fp = fn = 0
    for a, b in itertools.combinations(ids, 2):
        same_gold, same_pred = bool(gold[a] & gold[b]), bool(pred[a] & pred[b])
        tp += same_gold and same_pred
        fp += same_pred and not same_gold
        fn += same_gold and not same_pred
    return {**prf(tp, fp, fn), "messages_in_pairs": len(ids)}


def usage_summary() -> dict:
    logs = LLMRequestLog.objects.all()
    ok = [x for x in logs.filter(status="succeeded").values_list("latency_ms", flat=True) if x is not None]
    lat = {"n": len(ok)}
    if ok:
        lat.update(mean_ms=round(statistics.fmean(ok), 1), min_ms=min(ok), max_ms=max(ok))
    if len(ok) >= 20:
        q = statistics.quantiles(ok, n=100, method="inclusive")
        lat.update(p50_ms=round(q[49], 1), p95_ms=round(q[94], 1))
    tok = logs.filter(status="succeeded").exclude(input_tokens=None)
    return {"requests": logs.count(), "by_status": dict(logs.values_list("status").annotate(n=Count("id"))),
            "by_purpose": dict(logs.values_list("purpose").annotate(n=Count("id"))), "latency": lat,
            "mean_input_tokens": round(statistics.fmean(tok.values_list("input_tokens", flat=True)), 1) if tok else None,
            "mean_output_tokens": round(statistics.fmean([x or 0 for x in tok.values_list("output_tokens", flat=True)]), 1) if tok else None}


def _llm_prediction(msg_ids) -> dict[int, MessageAnalysis]:
    out = {}
    qs = (MessageAnalysis.objects.filter(message__source_message_id__in=msg_ids, source="llm", is_mock=False)
          .exclude(status="failed").select_related("message").order_by("message_id", "-created_at"))
    for a in qs:
        out.setdefault(a.message.source_message_id, a)
    return out


def evaluate(gold_items: list[GoldItem], allow_mock: bool = False) -> dict:
    ids = [g.tweet_id for g in gold_items]
    msgs = {m.source_message_id: m for m in CustomerMessage.objects.filter(source_message_id__in=ids)}
    mock_current = set(MessageAnalysis.objects.filter(message__source_message_id__in=ids, is_current=True)
                       .filter(source="mock").values_list("message__source_message_id", flat=True)) | \
        set(MessageAnalysis.objects.filter(message__source_message_id__in=ids, is_current=True, is_mock=True)
            .values_list("message__source_message_id", flat=True))
    analyses = _llm_prediction(ids)
    if allow_mock:  # pipeline smoke test only: fall back to current (mock) analyses, clearly labelled
        for a in MessageAnalysis.objects.filter(message__source_message_id__in=ids, is_current=True).select_related("message"):
            analyses.setdefault(a.message.source_message_id, a)
    elif not analyses and mock_current:
        return {"status": "refused", "reason": f"{len(mock_current)} evaluated messages only have MOCK analyses; "
                "mock output is not AI evaluation evidence. Analyse the evaluation sample with the real provider."}
    missing = [i for i in ids if i not in analyses]
    used = [g for g in gold_items if g.tweet_id in analyses]
    mock_used = [g.tweet_id for g in used if analyses[g.tweet_id].is_mock or analyses[g.tweet_id].source == "mock"]
    providers = Counter(f"{analyses[g.tweet_id].provider}/{analyses[g.tweet_id].model_name}/"
                        f"{analyses[g.tweet_id].prompt_version}" for g in used)

    pred_type, pred_keys, hybrid_keys, review = {}, {}, {}, 0
    for g in used:
        a = analyses[g.tweet_id]
        if a.status == MessageAnalysis.Status.NEEDS_REVIEW:
            review += 1
        pred_type[g.tweet_id] = a.message_type or "other"
        mentions = list(IssueMention.objects.filter(analysis=a).select_related("issue", "original_issue"))
        # baseline = issue the model chose; hybrid = after accepted regrouping (active mentions only)
        pred_keys[g.tweet_id] = {(m.original_issue or m.issue).key for m in mentions}
        # a mention deactivated by an accepted merge no longer contributes its old key
        hybrid_keys[g.tweet_id] = {m.issue.key for m in mentions if not (m.regrouped_at and not m.is_active)}
    gold_type = {g.tweet_id: g.message_type for g in used}
    gold_keys = {g.tweet_id: g.issue_keys for g in used}

    tp = sum(1 for t in gold_type if gold_type[t] == "complaint" and pred_type[t] == "complaint")
    fp = sum(1 for t in gold_type if gold_type[t] != "complaint" and pred_type[t] == "complaint")
    fn = sum(1 for t in gold_type if gold_type[t] == "complaint" and pred_type[t] != "complaint")
    labels = list(taxonomy.MESSAGE_TYPES)
    confusion = {g: {p: 0 for p in labels} for g in labels}
    for t in gold_type:
        confusion[gold_type[t]][pred_type[t]] += 1
    per_type = {}
    for lab in labels:
        ltp = confusion[lab][lab]
        lfp = sum(confusion[g][lab] for g in labels if g != lab)
        lfn = sum(confusion[lab][p] for p in labels if p != lab)
        per_type[lab] = prf(ltp, lfp, lfn)
    f1s = [v["f1"] for v in per_type.values() if v["support"] and v["f1"] is not None]
    accuracy = sum(confusion[lab][lab] for lab in labels) / len(used) if used else None
    cat = lambda keys: {k.split(".")[0] for k in keys}  # noqa: E731
    gp = [g.tweet_id for g in used if g.issue_keys or pred_keys[g.tweet_id] or hybrid_keys[g.tweet_id]]

    errors = []
    for g in used:
        t = g.tweet_id
        if gold_type[t] != pred_type[t] or gold_keys[t] != pred_keys[t]:
            errors.append({"tweet_id": t, "gold_type": gold_type[t], "pred_type": pred_type[t],
                           "gold_keys": ";".join(sorted(gold_keys[t])), "pred_keys": ";".join(sorted(pred_keys[t])),
                           "analysis_status": analyses[t].status,
                           "review_reasons": "; ".join(analyses[t].review_reasons),
                           "text": msgs[t].text if t in msgs else ""})
    return {
        "status": "measured" if not mock_used else "measured_with_MOCK_provider_NOT_AI_EVALUATION",
        "gold_items": len(gold_items), "evaluated": len(used), "missing_analysis": len(missing),
        "needs_review_predictions": review, "providers": dict(providers),
        "complaint_detection": prf(tp, fp, fn),
        "message_type": {"accuracy": _r(accuracy), "macro_f1": _r(sum(f1s) / len(f1s)) if f1s else None,
                         "per_class": per_type, "confusion_matrix": {"rows_gold_cols_pred": confusion}},
        "category_extraction": multilabel({t: cat(k) for t, k in gold_keys.items()},
                                          {t: cat(k) for t, k in pred_keys.items()}),
        "issue_extraction": multilabel(gold_keys, pred_keys),
        "grouping_pairwise": pairwise(gp, gold_keys, pred_keys),
        "grouping_pairwise_hybrid": pairwise(gp, gold_keys, hybrid_keys),
        "real_usage": usage_summary(),
        "errors": errors,
    }

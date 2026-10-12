from datetime import datetime, timezone

import pytest

from feedback.models import MessageAnalysis
from feedback.services.evaluation import GoldFileError, GoldItem, evaluate, load_gold, prf
from tests import factories

pytestmark = pytest.mark.django_db
T = datetime(2017, 10, 10, tzinfo=timezone.utc)
CHARGE, CRASH = "billing_subscription.unexpected_charge", "application_errors.app_crash"


def test_prf_math_and_zero_division():
    assert prf(8, 2, 2) == {"precision": 0.8, "recall": 0.8, "f1": 0.8, "tp": 8, "fp": 2, "fn": 2, "support": 10}
    assert prf(0, 0, 0)["precision"] is None
    assert prf(0, 3, 2)["f1"] == 0.0


def _predict(keys, mtype=None, mock=False):
    m = factories.analysed_message(factories.batch(), T, keys)
    a = m.analyses.get()
    a.is_mock = mock
    if mtype:
        a.message_type = mtype
    a.save()
    return m.source_message_id


def test_evaluate_metrics_against_gold():
    a = _predict((CHARGE,))            # correct
    b = _predict((CHARGE, CRASH))      # one extra key
    c = _predict((), mtype="question")  # missed complaint
    d = _predict((CRASH,))             # gold says praise -> false positive complaint
    gold = [GoldItem(a, "complaint", {CHARGE}), GoldItem(b, "complaint", {CHARGE}),
            GoldItem(c, "complaint", {CRASH}), GoldItem(d, "praise", set()), GoldItem(999, "question", set())]
    r = evaluate(gold)
    assert r["evaluated"] == 4 and r["missing_analysis"] == 1
    assert r["complaint_detection"] == prf(2, 1, 1)
    assert r["message_type"]["confusion_matrix"]["rows_gold_cols_pred"]["complaint"]["question"] == 1
    assert r["issue_extraction"]["micro"] == prf(2, 2, 1)
    # pairs among a,b,c,d: gold-same = (a,b); pred-same = (a,b), (b,d)
    assert r["grouping_pairwise"]["tp"] == 1 and r["grouping_pairwise"]["fp"] == 1 and r["grouping_pairwise"]["fn"] == 0
    assert len(r["errors"]) == 3


def test_mock_predictions_are_refused():
    a = _predict((CHARGE,), mock=True)
    assert evaluate([GoldItem(a, "complaint", {CHARGE})])["status"] == "refused"
    assert "NOT_AI" in evaluate([GoldItem(a, "complaint", {CHARGE})], allow_mock=True)["status"]


def test_gold_file_validation(tmp_path):
    p = tmp_path / "g.csv"
    p.write_text("tweet_id,message_type,issue_keys\n1,complaint,billing_subscription.refund_request\n2,,\n",
                 encoding="utf-8")
    assert [g.tweet_id for g in load_gold(p)] == [1]  # unlabeled rows skipped
    p.write_text("tweet_id,message_type,issue_keys\n1,complaint,made.up\n", encoding="utf-8")
    with pytest.raises(GoldFileError):
        load_gold(p)
    with pytest.raises(GoldFileError):
        load_gold(tmp_path / "missing.csv")


def test_needs_review_counts_as_no_aspects():
    a = _predict((CHARGE,))
    MessageAnalysis.objects.filter(message__source_message_id=a).update(status="needs_review")
    assert evaluate([GoldItem(a, "complaint", {CHARGE})])["needs_review_predictions"] == 1

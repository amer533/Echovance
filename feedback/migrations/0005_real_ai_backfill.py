"""Backfill provenance for data that existed before the real-AI upgrade. Nothing is deleted.

* MessageAnalysis.source: is_mock -> "mock", otherwise "llm".
* IssueMention.source: copied from its analysis.
* ImportBatch.data_classification: "public" (all pre-existing batches are the published Kaggle dataset).
* IssueAlert: data_scope "legacy_mixed" (computed over every active mention, mock and real) and an
  input fingerprint over the active mentions of its window (no analysis ran after these alerts were
  created, so this equals their original inputs).
"""
import hashlib

from django.db import migrations
from django.db.models import OuterRef, Subquery


def fingerprint(ids):
    return hashlib.sha256(",".join(str(i) for i in sorted(ids)).encode()).hexdigest()


def forwards(apps, schema_editor):
    MessageAnalysis = apps.get_model("feedback", "MessageAnalysis")
    IssueMention = apps.get_model("feedback", "IssueMention")
    ImportBatch = apps.get_model("feedback", "ImportBatch")
    IssueAlert = apps.get_model("feedback", "IssueAlert")

    MessageAnalysis.objects.filter(is_mock=True).update(source="mock")
    MessageAnalysis.objects.filter(is_mock=False).update(source="llm")
    IssueMention.objects.update(source=Subquery(
        MessageAnalysis.objects.filter(pk=OuterRef("analysis_id")).values("source")[:1]))
    ImportBatch.objects.update(data_classification="public")
    for alert in IssueAlert.objects.all():
        ids = IssueMention.objects.filter(issue_id=alert.issue_id, is_active=True,
                                          message_created_at__gte=alert.window_start,
                                          message_created_at__lt=alert.window_end).values_list("id", flat=True)
        alert.data_scope = "legacy_mixed"
        alert.input_fingerprint = fingerprint(ids)
        alert.save(update_fields=["data_scope", "input_fingerprint"])


class Migration(migrations.Migration):
    dependencies = [("feedback", "0004_real_ai_schema")]
    operations = [migrations.RunPython(forwards, migrations.RunPython.noop)]

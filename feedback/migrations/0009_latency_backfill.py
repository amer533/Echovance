"""Additive backfill of the latency breakdown for real analyses recorded before the fix.

Before 0008, MessageAnalysis.latency_ms included budget admission / rate-limiter waiting. It is NOT
changed. For analyses whose request can be matched to its LLMRequestLog row by request fingerprint
(the pilot job), we fill the new fields from the request log (the provider-only time measured there):
  provider_latency_ms = log.latency_ms ; total_ms = latency_ms ; queue_wait_ms = total - provider
(queue_wait_ms therefore also contains small client overhead). Unmatched analyses (the first three
calls, made before fingerprints existed) keep NULL in the new fields rather than a guessed value.
"""
from django.db import migrations


def forwards(apps, schema_editor):
    MessageAnalysis = apps.get_model("feedback", "MessageAnalysis")
    LLMRequestLog = apps.get_model("feedback", "LLMRequestLog")
    for a in MessageAnalysis.objects.filter(source="llm", provider_latency_ms__isnull=True).exclude(input_fingerprint=""):
        log = LLMRequestLog.objects.filter(request_fingerprint=a.input_fingerprint, status="succeeded",
                                           purpose="message_extraction").order_by("created_at").first()
        if log is None or log.latency_ms is None or a.latency_ms is None:
            continue
        a.provider_latency_ms = log.latency_ms
        a.total_ms = a.latency_ms
        a.queue_wait_ms = max(0, a.latency_ms - log.latency_ms)
        a.save(update_fields=["provider_latency_ms", "total_ms", "queue_wait_ms"])


class Migration(migrations.Migration):
    dependencies = [("feedback", "0008_latency_breakdown")]
    operations = [migrations.RunPython(forwards, migrations.RunPython.noop)]

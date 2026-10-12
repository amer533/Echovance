from django import forms
from django.conf import settings
from django.contrib.auth import get_user_model

from feedback import taxonomy
from feedback.models import ActionItem, AnalysisJob, ImportBatch, Issue, IssueAlert

ISSUE_KEY_CHOICES = [(taxonomy.issue_key(c, sub), f"{d['label']}: {desc}")
                     for c, d in taxonomy.CATEGORIES.items() for sub, desc in d["subtypes"].items()]


class ImportForm(forms.Form):
    file = forms.FileField(help_text="UTF-8 CSV in the twcs.csv schema (tweet_id, author_id, inbound, "
                                     "created_at, text, response_tweet_id, in_response_to_tweet_id).")
    brand_account = forms.CharField(max_length=64, initial="SpotifyCares",
                                    help_text="Brand support handle; its outbound tweets are stored as agent replies.")
    data_classification = forms.ChoiceField(
        choices=ImportBatch.DataClass.choices, initial=ImportBatch.DataClass.INTERNAL, required=False,
        help_text="Privacy gate: OpenAI may receive 'public' and 'internal' data (LLM_PAID_ALLOWED_DATA_CLASSES); "
                  "confidential data is never sent to an external AI provider.")

    def clean_data_classification(self):
        # Missing value -> "internal" (safe default, not "public").
        return self.cleaned_data.get("data_classification") or ImportBatch.DataClass.INTERNAL

    def clean_file(self):
        f = self.cleaned_data["file"]
        if f.size > settings.CSV_MAX_UPLOAD_BYTES:
            raise forms.ValidationError(f"File is larger than {settings.CSV_MAX_UPLOAD_BYTES // 1024} KB.")
        return f


class RunAnalysisForm(forms.Form):
    limit = forms.IntegerField(min_value=1, max_value=settings.ANALYSIS_MAX_BATCH_PER_REQUEST, initial=10)


class DetectForm(forms.Form):
    window_end = forms.DateField(required=False, widget=forms.DateInput(attrs={"type": "date"}),
                                 help_text="Exclusive end date (UTC). Leave empty for the latest data.")
    all_windows = forms.BooleanField(required=False, label="Backfill every historical window")
    scope = forms.ChoiceField(choices=[("real", "Real AI + human analyses"), ("demo", "Mock demo analyses")],
                              initial="real", required=False)

    def clean_scope(self):
        return self.cleaned_data.get("scope") or "real"


class AlertReviewForm(forms.Form):
    status = forms.ChoiceField(choices=[(IssueAlert.Status.ACKNOWLEDGED, "Acknowledge"),
                                        (IssueAlert.Status.DISMISSED, "Dismiss")])
    note = forms.CharField(max_length=500, required=False, widget=forms.Textarea(attrs={"rows": 2}))


class IssueStatusForm(forms.Form):
    status = forms.ChoiceField(choices=Issue.Status.choices)


class ActionCreateForm(forms.Form):
    title = forms.CharField(max_length=160)
    description = forms.CharField(max_length=2000, widget=forms.Textarea(attrs={"rows": 3}))
    rationale = forms.CharField(max_length=1000, required=False, widget=forms.Textarea(attrs={"rows": 2}))


class TransitionForm(forms.Form):
    status = forms.ChoiceField(choices=ActionItem.Status.choices)
    note = forms.CharField(max_length=1000, required=False, widget=forms.Textarea(attrs={"rows": 2}))


class AssignForm(forms.Form):
    assignee = forms.ModelChoiceField(queryset=get_user_model().objects.filter(is_active=True).order_by("username"))
    due_date = forms.DateField(required=False, widget=forms.DateInput(attrs={"type": "date"}))


class EditProposalForm(forms.Form):
    title = forms.CharField(max_length=160)
    description = forms.CharField(max_length=2000, widget=forms.Textarea(attrs={"rows": 4}))


class CorrectionForm(forms.Form):
    """Human correction: message type plus up to three issue aspects with verbatim evidence."""
    message_type = forms.ChoiceField(choices=[(k, k.replace("_", " ")) for k in taxonomy.MESSAGE_TYPES])
    note = forms.CharField(max_length=200, required=False)
    ROWS = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for i in range(1, self.ROWS + 1):
            self.fields[f"aspect{i}_key"] = forms.ChoiceField(choices=[("", "—")] + ISSUE_KEY_CHOICES, required=False,
                                                              label=f"Issue {i}")
            self.fields[f"aspect{i}_sentiment"] = forms.ChoiceField(
                choices=[(x, x) for x in taxonomy.SENTIMENTS], initial="negative", required=False, label="Sentiment")
            self.fields[f"aspect{i}_evidence"] = forms.CharField(max_length=300, required=False,
                                                                 label="Evidence (copy exactly from the message)")

    def clean(self):
        data = super().clean()
        for i in range(1, self.ROWS + 1):
            if bool(data.get(f"aspect{i}_key")) != bool((data.get(f"aspect{i}_evidence") or "").strip()):
                raise forms.ValidationError(f"Issue {i} needs both an issue and its evidence quote.")
        return data

    def aspects(self) -> list[dict]:
        out = []
        for i in range(1, self.ROWS + 1):
            key = self.cleaned_data.get(f"aspect{i}_key")
            if key:
                cat, sub = key.split(".")
                out.append({"category": cat, "subtype": sub, "evidence": self.cleaned_data[f"aspect{i}_evidence"],
                            "sentiment": self.cleaned_data.get(f"aspect{i}_sentiment") or "negative",
                            "summary": "Corrected by a reviewer."})
        return out


class JobForm(forms.Form):
    selection = forms.ChoiceField(choices=AnalysisJob.Selection.choices)
    max_messages = forms.IntegerField(min_value=1, max_value=10_000, initial=settings.JOBS["DEFAULT_MAX_MESSAGES"])
    message_ids = forms.CharField(required=False, help_text="Only for 'Specific conversations': comma-separated "
                                                             "message ids (as shown in message URLs).")
    max_requests = forms.IntegerField(min_value=1, required=False,
                                      help_text="Default: one request per message (no retries).")
    max_cost_usd = forms.DecimalField(min_value=0, max_digits=10, decimal_places=4, required=False,
                                      label="Max cost (USD)",
                                      help_text="Default: the estimated cost of the job's requests.")

    def parsed_message_ids(self) -> list[int]:
        raw = self.cleaned_data.get("message_ids") or ""
        return [int(x) for x in raw.replace(" ", "").split(",") if x.isdigit()]


class LabelForm(forms.Form):
    message_type = forms.ChoiceField(choices=[(k, k.replace("_", " ")) for k in taxonomy.MESSAGE_TYPES],
                                     widget=forms.RadioSelect)
    issue_keys = forms.MultipleChoiceField(choices=ISSUE_KEY_CHOICES, required=False,
                                           widget=forms.SelectMultiple(attrs={"size": 10}))
    notes = forms.CharField(max_length=500, required=False, widget=forms.Textarea(attrs={"rows": 2}))

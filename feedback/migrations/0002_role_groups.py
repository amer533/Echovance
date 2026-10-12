from django.contrib.auth.management import create_permissions
from django.db import migrations

ROLES = {
    "Managers": ["manage_actions", "run_pipeline"],
    "Analysts": ["run_pipeline"],
    "Viewers": [],
}


def create_groups(apps, schema_editor):
    # Permissions are normally created after migrations; create them now so groups can reference them.
    app_config = apps.get_app_config("feedback")
    app_config.models_module = True
    create_permissions(app_config, apps=apps, verbosity=0)
    app_config.models_module = None
    Group = apps.get_model("auth", "Group")
    Permission = apps.get_model("auth", "Permission")
    for name, codenames in ROLES.items():
        group, _ = Group.objects.get_or_create(name=name)
        group.permissions.set(Permission.objects.filter(content_type__app_label="feedback", codename__in=codenames))


def remove_groups(apps, schema_editor):
    apps.get_model("auth", "Group").objects.filter(name__in=ROLES).delete()


class Migration(migrations.Migration):
    dependencies = [("feedback", "0001_initial"), ("auth", "0012_alter_user_first_name_max_length"),
                    ("contenttypes", "0002_remove_content_type_name")]
    operations = [migrations.RunPython(create_groups, remove_groups)]

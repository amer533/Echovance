import secrets

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.core.management.base import BaseCommand

USERS = [("manager", "Managers"), ("analyst", "Analysts"), ("viewer", "Viewers"), ("agent1", "Viewers")]


class Command(BaseCommand):
    help = "Create demo users (manager, analyst, viewer, agent1). Existing users are left unchanged."

    def add_arguments(self, parser):
        parser.add_argument("--password", help="Password for all demo users (default: random, printed once)")

    def handle(self, password, **opts):
        User = get_user_model()
        password = password or secrets.token_urlsafe(9)
        for username, group in USERS:
            user, created = User.objects.get_or_create(username=username)
            if created:
                user.set_password(password)
                user.save()
            user.groups.add(Group.objects.get(name=group))
            self.stdout.write(f"{username:8} group={group:9} {'created' if created else 'exists (password unchanged)'}")
        self.stdout.write(f"Password for newly created users: {password}")

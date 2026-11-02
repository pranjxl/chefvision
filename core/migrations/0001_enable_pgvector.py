from django.db import migrations
from pgvector.django import VectorExtension


class Migration(migrations.Migration):
    """Enable the pgvector extension so later phases can store recipe embeddings."""

    dependencies = []

    operations = [VectorExtension()]

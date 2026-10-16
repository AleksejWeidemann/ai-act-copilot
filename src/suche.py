"""Wiederverwendbare Hybrid-Suche (Volltext + Vektor) im Index "aiact".

Genutzt von search_cli.py und rag.py. Konfiguration kommt aus Umgebungsvariablen.
"""

import time
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

from build_index import INDEX_NAME
from smoke_test import env

TOP = 5
FELDER = ["typ", "nummer", "titel", "chunk_nr"]


def aiact_search_client() -> SearchClient:
    """Erzeugt den SearchClient für den Index "aiact" aus Umgebungsvariablen."""
    return SearchClient(
        endpoint=env("AZURE_SEARCH_ENDPOINT"),
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(env("AZURE_SEARCH_API_KEY")),
    )


def bette_frage_ein(client: Any, frage: str) -> tuple[list[float], int]:
    """Bettet die Frage ein, loggt Tokens und Latenz, liefert Vektor und Tokenzahl."""
    beginn = time.perf_counter()
    antwort = client.embeddings.create(model=env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"), input=frage)
    dauer_ms = (time.perf_counter() - beginn) * 1000
    tokens = antwort.usage.prompt_tokens
    print(f"Embedding: {tokens} Tokens | {dauer_ms:.0f} ms")
    return antwort.data[0].embedding, tokens


def hybrid_suche(
    search_client: Any, frage: str, vektor: list[float], felder: list[str] = FELDER
) -> list[dict]:
    """Kombiniert Volltext- und Vektorsuche (RRF) und liefert die Top-5-Treffer mit `felder`."""
    beginn = time.perf_counter()
    treffer = list(
        search_client.search(
            search_text=frage,
            vector_queries=[VectorizedQuery(vector=vektor, k_nearest_neighbors=TOP, fields="vector")],
            select=felder,
            top=TOP,
        )
    )
    dauer_ms = (time.perf_counter() - beginn) * 1000
    print(f"Suche: {len(treffer)} Treffer | {dauer_ms:.0f} ms")
    return treffer

"""Hybrid-Suche (Volltext + Vektor) im Index "aiact", Top 5.

Aufruf: python src/search_cli.py "biometrische Echtzeit-Fernidentifizierung"
"""

import sys
import time

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv
from openai import AzureOpenAI

from build_index import INDEX_NAME
from smoke_test import bereinige, env, openai_client

TOP = 5
FELDER = ["typ", "nummer", "titel", "chunk_nr"]


def bette_frage_ein(client: AzureOpenAI, frage: str) -> list[float]:
    """Bettet die Suchanfrage ein und loggt Tokens und Latenz."""
    beginn = time.perf_counter()
    antwort = client.embeddings.create(model=env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"), input=frage)
    dauer_ms = (time.perf_counter() - beginn) * 1000
    print(f"Embedding: {antwort.usage.prompt_tokens} Tokens | {dauer_ms:.0f} ms")
    return antwort.data[0].embedding


def hybrid_suche(search_client: SearchClient, frage: str, vektor: list[float]) -> list[dict]:
    """Kombiniert Volltextsuche und Vektorsuche (RRF) und liefert die Top-5-Treffer."""
    beginn = time.perf_counter()
    treffer = list(
        search_client.search(
            search_text=frage,
            vector_queries=[VectorizedQuery(vector=vektor, k_nearest_neighbors=TOP, fields="vector")],
            select=FELDER,
            top=TOP,
        )
    )
    dauer_ms = (time.perf_counter() - beginn) * 1000
    print(f"Suche: {len(treffer)} Treffer | {dauer_ms:.0f} ms")
    return treffer


def formatiere(rang: int, treffer: dict) -> str:
    """Eine Ausgabezeile: Rang, typ, nummer, titel, chunk_nr, score."""
    return (
        f"{rang}. {treffer['typ']:<8} {treffer['nummer']:<5} {treffer['titel'][:60]:<60} "
        f"chunk {treffer['chunk_nr']:<2} score {treffer['@search.score']:.4f}"
    )


def main() -> int:
    """Liest die Frage aus der Kommandozeile und gibt die Top-5-Treffer aus."""
    if len(sys.argv) < 2:
        print('Aufruf: python src/search_cli.py "Suchanfrage"')
        return 2
    frage = " ".join(sys.argv[1:])
    load_dotenv()
    try:
        vektor = bette_frage_ein(openai_client(), frage)
        search_client = SearchClient(
            endpoint=env("AZURE_SEARCH_ENDPOINT"),
            index_name=INDEX_NAME,
            credential=AzureKeyCredential(env("AZURE_SEARCH_API_KEY")),
        )
        treffer = hybrid_suche(search_client, frage, vektor)
    except Exception as exc:
        print(f"FEHLER {type(exc).__name__}: {bereinige(str(exc))[:300]}")
        return 1
    for rang, t in enumerate(treffer, start=1):
        print(formatiere(rang, t))
    return 0


if __name__ == "__main__":
    sys.exit(main())

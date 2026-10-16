"""Lädt data/aiact/aiact.json als Chunks mit Embeddings in den Azure-AI-Search-Index "aiact".

Aufruf: python src/build_index.py
Der Index wird bei jedem Lauf gelöscht und neu angelegt (wiederholbar).
Konfiguration kommt aus .env; Keys und Endpunkte werden nie ausgegeben.
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import openai
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    LexicalAnalyzerName,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from dotenv import load_dotenv
from openai import AzureOpenAI

from smoke_test import bereinige, env

JSON_PFAD = Path(__file__).resolve().parent.parent / "data" / "aiact" / "aiact.json"
INDEX_NAME = "aiact"
MAX_ZEICHEN = 2000
DIMENSIONEN = 1536
EMBEDDING_BATCH = 16
UPLOAD_BATCH = 100
MAX_VERSUCHE = 6
METADATEN = ("typ", "nummer", "titel", "geaendert_durch_omnibus", "fassung", "url")

# Trennstellen von grob nach fein. Jedes Muster markiert das Ende eines Stücks;
# der Trenner bleibt am Stück hängen, damit "".join(chunks) den Originaltext ergibt.
TRENNER = (
    r"\n",  # Zeilenende = Absatz bzw. Nummer
    r"\s+(?=\(\d+[a-z]?\)\s)",  # vor "(2) " innerhalb einer Zeile
    r"(?<=[.;:])\s+",  # Satzende
    r"\s+",  # Wortgrenze
)


def schneide_nach(text: str, muster: str) -> list[str]:
    """Teilt den Text hinter jedem Treffer des Musters; der Treffer bleibt am linken Stück."""
    grenzen = [m.end() for m in re.finditer(muster, text)]
    anfaenge = [0, *grenzen]
    enden = [*grenzen, len(text)]
    return [text[a:b] for a, b in zip(anfaenge, enden) if a < b]


def zerlege(text: str, max_zeichen: int, ebene: int = 0) -> list[str]:
    """Zerlegt Text in Stücke bis max_zeichen, mit der gröbsten passenden Trennstelle."""
    if len(text) <= max_zeichen:
        return [text]
    if ebene == len(TRENNER):
        return [text[i : i + max_zeichen] for i in range(0, len(text), max_zeichen)]
    stuecke = []
    for stueck in schneide_nach(text, TRENNER[ebene]):
        stuecke.extend(zerlege(stueck, max_zeichen, ebene + 1))
    return stuecke


def packe(stuecke: list[str], max_zeichen: int) -> list[str]:
    """Fasst aufeinanderfolgende Stücke gierig zu Chunks bis max_zeichen zusammen."""
    chunks: list[str] = []
    for stueck in stuecke:
        if chunks and len(chunks[-1]) + len(stueck) <= max_zeichen:
            chunks[-1] += stueck
        else:
            chunks.append(stueck)
    return chunks


def chunke_eintrag(eintrag: dict, max_zeichen: int = MAX_ZEICHEN) -> list[dict]:
    """Teilt einen Artikel/Anhang in Chunks; jeder Chunk behält die Metadaten."""
    texte = packe(zerlege(eintrag["text"], max_zeichen), max_zeichen)
    return [
        {
            "id": f"{eintrag['typ']}_{eintrag['nummer']}_{nr}",
            **{feld: eintrag[feld] for feld in METADATEN},
            "chunk_nr": nr,
            "text": text,
        }
        for nr, text in enumerate(texte)
    ]


def embedding_eingabe(chunk: dict) -> str:
    """Text fürs Embedding: Kopfzeile mit Fundstelle und Titel, dann der Chunk-Text."""
    art = "Artikel" if chunk["typ"] == "artikel" else "Anhang"
    return f"{art} {chunk['nummer']} – {chunk['titel']}\n{chunk['text']}"


def bette_ein_mit_retry(client: Any, deployment: str, texte: list[str]) -> tuple[list[list[float]], int]:
    """Bettet eine Liste von Texten ein; wartet bei 429 exponentiell und versucht es erneut."""
    for versuch in range(MAX_VERSUCHE):
        try:
            antwort = client.embeddings.create(model=deployment, input=texte)
            return [d.embedding for d in antwort.data], antwort.usage.prompt_tokens
        except openai.RateLimitError:
            if versuch == MAX_VERSUCHE - 1:
                raise
            wartezeit = 2**versuch
            print(f"  429 – warte {wartezeit} s (Versuch {versuch + 1}/{MAX_VERSUCHE})")
            time.sleep(wartezeit)
    raise AssertionError("nicht erreichbar")


def bette_alle_ein(client: Any, deployment: str, chunks: list[dict]) -> int:
    """Ergänzt jeden Chunk um das Feld 'vector'; liefert die verbrauchten Tokens."""
    tokens_gesamt = 0
    for start in range(0, len(chunks), EMBEDDING_BATCH):
        batch = chunks[start : start + EMBEDDING_BATCH]
        beginn = time.perf_counter()
        try:
            vektoren, tokens = bette_ein_mit_retry(client, deployment, [embedding_eingabe(c) for c in batch])
        except Exception as exc:
            raise RuntimeError(f"Embedding fehlgeschlagen: {bereinige(str(exc))[:300]}") from None
        for chunk, vektor in zip(batch, vektoren):
            if len(vektor) != DIMENSIONEN:
                raise RuntimeError(f"Embedding hat {len(vektor)} statt {DIMENSIONEN} Dimensionen.")
            chunk["vector"] = vektor
        tokens_gesamt += tokens
        dauer_ms = (time.perf_counter() - beginn) * 1000
        print(f"  Embeddings {start + len(batch):>4}/{len(chunks)} | {tokens:>5} Tokens | {dauer_ms:6.0f} ms")
    return tokens_gesamt


def index_definition() -> SearchIndex:
    """Beschreibt den Index: Metadaten, deutscher Volltext und 1536-dim. HNSW-Vektor."""
    felder = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="typ", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="nummer", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="titel", analyzer_name=LexicalAnalyzerName.DE_MICROSOFT),
        SimpleField(name="chunk_nr", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SearchableField(name="text", analyzer_name=LexicalAnalyzerName.DE_MICROSOFT),
        SimpleField(name="geaendert_durch_omnibus", type=SearchFieldDataType.Boolean, filterable=True),
        SimpleField(name="fassung", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="url", type=SearchFieldDataType.String),
        SearchField(
            name="vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=DIMENSIONEN,
            vector_search_profile_name="hnsw-profil",
        ),
    ]
    vektorsuche = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
        profiles=[VectorSearchProfile(name="hnsw-profil", algorithm_configuration_name="hnsw")],
    )
    return SearchIndex(name=INDEX_NAME, fields=felder, vector_search=vektorsuche)


def lege_index_neu_an(index_client: SearchIndexClient) -> None:
    """Löscht den Index "aiact" (falls vorhanden) und legt ihn neu an."""
    try:
        index_client.delete_index(INDEX_NAME)
        print(f"Index '{INDEX_NAME}' gelöscht.")
    except ResourceNotFoundError:
        pass
    index_client.create_index(index_definition())
    print(f"Index '{INDEX_NAME}' angelegt.")


def lade_hoch(search_client: SearchClient, chunks: list[dict]) -> None:
    """Lädt die Chunks in Batches hoch und bricht ab, wenn ein Dokument scheitert."""
    for start in range(0, len(chunks), UPLOAD_BATCH):
        batch = chunks[start : start + UPLOAD_BATCH]
        beginn = time.perf_counter()
        ergebnisse = search_client.upload_documents(documents=batch)
        fehler = [r.key for r in ergebnisse if not r.succeeded]
        if fehler:
            raise RuntimeError(f"Upload fehlgeschlagen für: {', '.join(fehler[:10])}")
        dauer_ms = (time.perf_counter() - beginn) * 1000
        print(f"  Upload {start + len(batch):>4}/{len(chunks)} | {dauer_ms:6.0f} ms")


def main() -> int:
    """Chunkt, bettet ein, legt den Index neu an, lädt hoch und gibt eine Bilanz aus."""
    load_dotenv()
    beginn = time.perf_counter()
    eintraege = json.loads(JSON_PFAD.read_text(encoding="utf-8"))
    chunks = [c for e in eintraege for c in chunke_eintrag(e)]
    print(f"{len(eintraege)} Einträge → {len(chunks)} Chunks")
    try:
        openai_client = AzureOpenAI(
            azure_endpoint=env("AZURE_OPENAI_ENDPOINT"),
            api_key=env("AZURE_OPENAI_API_KEY"),
            api_version=env("AZURE_OPENAI_API_VERSION"),
            max_retries=0,  # Retry bei 429 übernimmt bette_ein_mit_retry
        )
        tokens = bette_alle_ein(openai_client, env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"), chunks)
        credential = AzureKeyCredential(env("AZURE_SEARCH_API_KEY"))
        endpoint = env("AZURE_SEARCH_ENDPOINT")
        lege_index_neu_an(SearchIndexClient(endpoint=endpoint, credential=credential))
        lade_hoch(SearchClient(endpoint=endpoint, index_name=INDEX_NAME, credential=credential), chunks)
    except Exception as exc:
        print(f"FEHLER {type(exc).__name__}: {bereinige(str(exc))[:300]}")
        return 1
    dauer = time.perf_counter() - beginn
    print(f"Fertig: {len(chunks)} Chunks | {tokens} Embedding-Tokens | {dauer:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

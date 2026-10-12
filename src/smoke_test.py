"""Smoke-Test: prüft Chat-Deployment, Embedding-Deployment und Azure AI Search.

Aufruf: python src/smoke_test.py
Konfiguration kommt aus .env (über python-dotenv). Keys und Endpunkte werden
niemals ausgegeben – auch nicht in Fehlermeldungen.
"""

import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from dotenv import load_dotenv
from openai import AzureOpenAI

SENSIBLE_VARIABLEN = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_SEARCH_ENDPOINT",
    "AZURE_SEARCH_API_KEY",
)
MAX_FEHLERLAENGE = 300


@dataclass
class Ergebnis:
    """Ergebnis einer einzelnen Prüfung."""

    name: str
    ok: bool
    latenz_ms: float
    details: str


def env(name: str) -> str:
    """Liest eine Umgebungsvariable oder wirft einen verständlichen Fehler."""
    wert = os.getenv(name)
    if not wert:
        raise RuntimeError(f"Umgebungsvariable {name} fehlt oder ist leer (siehe .env.example).")
    return wert


def bereinige(text: str) -> str:
    """Entfernt Keys, Endpunkte und deren Hostnamen aus einem Text."""
    for name in SENSIBLE_VARIABLEN:
        wert = os.getenv(name)
        if not wert:
            continue
        kandidaten = [wert, wert.rstrip("/")]
        host = urlparse(wert).hostname
        if host:
            kandidaten.append(host)
            kandidaten.append(host.split(".")[0])
        for kandidat in kandidaten:
            if kandidat:
                text = text.replace(kandidat, "***")
    return text


def openai_client() -> AzureOpenAI:
    """Erzeugt den AzureOpenAI-Client aus Umgebungsvariablen."""
    return AzureOpenAI(
        azure_endpoint=env("AZURE_OPENAI_ENDPOINT"),
        api_key=env("AZURE_OPENAI_API_KEY"),
        api_version=env("AZURE_OPENAI_API_VERSION"),
    )


def search_client() -> SearchIndexClient:
    """Erzeugt den SearchIndexClient für Azure AI Search aus Umgebungsvariablen."""
    return SearchIndexClient(
        endpoint=env("AZURE_SEARCH_ENDPOINT"),
        credential=AzureKeyCredential(env("AZURE_SEARCH_API_KEY")),
    )


def pruefe_chat(client: Any, deployment: str) -> str:
    """Stellt dem Chat-Deployment eine kurze Testfrage, liefert Tokens und Antwort."""
    antwort = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "user", "content": "Antworte in genau einem kurzen Satz: Hörst du mich?"}
        ],
    )
    text = (antwort.choices[0].message.content or "").strip().replace("\n", " ")
    usage = antwort.usage
    return f"Tokens: {usage.prompt_tokens} ein / {usage.completion_tokens} aus | Antwort: {text}"


def pruefe_embedding(client: Any, deployment: str) -> str:
    """Bettet 'Hochrisiko-KI-System' ein, liefert Vektorlänge und Tokens."""
    antwort = client.embeddings.create(model=deployment, input="Hochrisiko-KI-System")
    laenge = len(antwort.data[0].embedding)
    return f"Tokens: {antwort.usage.prompt_tokens} | Vektorlänge: {laenge}"


def pruefe_suche(index_client: Any) -> str:
    """Listet die vorhandenen Indizes in Azure AI Search auf."""
    namen = list(index_client.list_index_names())
    liste = f" ({', '.join(namen)})" if namen else ""
    return f"{len(namen)} Indizes{liste}"


def fuehre_pruefungen_aus(pruefungen: list[tuple[str, Callable[[], str]]]) -> list[Ergebnis]:
    """Führt alle Prüfungen nacheinander aus; ein Fehler stoppt die übrigen nicht."""
    ergebnisse = []
    for name, pruefung in pruefungen:
        start = time.perf_counter()
        try:
            details = pruefung()
            ok = True
        except Exception as exc:  # bewusst breit: jede Prüfung soll weiterlaufen
            details = f"{type(exc).__name__}: {bereinige(str(exc))[:MAX_FEHLERLAENGE]}"
            ok = False
        latenz_ms = (time.perf_counter() - start) * 1000
        ergebnisse.append(Ergebnis(name, ok, latenz_ms, details))
    return ergebnisse


def formatiere(ergebnis: Ergebnis) -> str:
    """Formatiert ein Ergebnis als eine Ausgabezeile ohne Geheimnisse."""
    status = "OK    " if ergebnis.ok else "FEHLER"
    return bereinige(
        f"[{status}] {ergebnis.name:<10} | {ergebnis.latenz_ms:7.0f} ms | {ergebnis.details}"
    )


def main() -> int:
    """Lädt .env, führt die drei Prüfungen aus und gibt 0 zurück, wenn alle OK sind."""
    load_dotenv()
    pruefungen = [
        ("Chat", lambda: pruefe_chat(openai_client(), env("AZURE_OPENAI_CHAT_DEPLOYMENT"))),
        (
            "Embedding",
            lambda: pruefe_embedding(openai_client(), env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")),
        ),
        ("Suche", lambda: pruefe_suche(search_client())),
    ]
    ergebnisse = fuehre_pruefungen_aus(pruefungen)
    for ergebnis in ergebnisse:
        print(formatiere(ergebnis))
    return 0 if all(e.ok for e in ergebnisse) else 1


if __name__ == "__main__":
    sys.exit(main())

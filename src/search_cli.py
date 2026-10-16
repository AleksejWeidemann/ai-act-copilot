"""Hybrid-Suche (Volltext + Vektor) im Index "aiact", Top 5.

Aufruf: python src/search_cli.py "biometrische Echtzeit-Fernidentifizierung"
"""

import sys

from dotenv import load_dotenv

from smoke_test import bereinige, openai_client
from suche import aiact_search_client, bette_frage_ein, hybrid_suche


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
        vektor, _ = bette_frage_ein(openai_client(), frage)
        treffer = hybrid_suche(aiact_search_client(), frage, vektor)
    except Exception as exc:
        print(f"FEHLER {type(exc).__name__}: {bereinige(str(exc))[:300]}")
        return 1
    for rang, t in enumerate(treffer, start=1):
        print(formatiere(rang, t))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Stellt eine Frage an die RAG-Kette und gibt Antwort, Quellen, Warnungen und Kosten aus.

Aufruf: python src/ask_cli.py "Ist Emotionserkennung am Arbeitsplatz erlaubt?"
"""

import sys

from dotenv import load_dotenv

from rag import Antwort, beantworte
from smoke_test import MAX_FEHLERLAENGE, bereinige

DISCLAIMER = "Keine Rechtsberatung – Entscheidungsunterstützung mit menschlicher Prüfung."


def formatiere_ergebnis(ergebnis: Antwort) -> str:
    """Formatiert Antwort, Quellen, Warnungen, Tokens und Latenz als Text."""
    zeilen = ["", ergebnis.antwort, "", "Quellen:"]
    for q in ergebnis.quellen:
        zeilen.append(f"  - {q['typ']} {q['nummer']} – {q['titel'][:60]} (chunk {q['chunk_nr']})")
    for warnung in ergebnis.warnungen:
        zeilen.append(f"WARNUNG: {warnung}")
    t, l = ergebnis.tokens, ergebnis.latenz
    zeilen.append(
        f"Tokens: {t['embedding']} Embedding / {t['prompt']} ein / {t['completion']} aus | "
        f"Latenz: {l['gesamt_ms']:.0f} ms (Embedding {l['embedding_ms']:.0f}, "
        f"Suche {l['suche_ms']:.0f}, Chat {l['chat_ms']:.0f})"
    )
    zeilen.append(DISCLAIMER)
    return "\n".join(zeilen)


def main() -> int:
    """Liest die Frage aus der Kommandozeile und gibt das Ergebnis der RAG-Kette aus."""
    if len(sys.argv) < 2:
        print('Aufruf: python src/ask_cli.py "Frage"')
        return 2
    load_dotenv()
    try:
        ergebnis = beantworte(" ".join(sys.argv[1:]))
    except Exception as exc:
        print(f"FEHLER {type(exc).__name__}: {bereinige(str(exc))[:MAX_FEHLERLAENGE]}")
        return 1
    print(formatiere_ergebnis(ergebnis))
    return 0


if __name__ == "__main__":
    sys.exit(main())

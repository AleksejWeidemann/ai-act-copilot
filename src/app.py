"""Streamlit-Oberfläche für den Frage-Modus: Zugangscode, Beispielfragen, Antwort mit Quellen.

Aufruf: streamlit run src/app.py
Ohne gesetzte Umgebungsvariable APP_ZUGANGSCODE startet die App nicht (fail closed).
"""

import hmac
import os

import streamlit as st
from dotenv import load_dotenv

import rag
from rag import ABLEHNUNG 
from ask_cli import DISCLAIMER
from smoke_test import MAX_FEHLERLAENGE, bereinige

MAX_FRAGEN = 20
BEISPIELFRAGEN = [
    "Ist Emotionserkennung am Arbeitsplatz erlaubt?",
    "Welche Pflichten haben Betreiber von Hochrisiko-KI-Systemen?",
    "Was muss die technische Dokumentation enthalten?",
]


def zugang_erlaubt(eingabe: str, code: str) -> bool:
    """Vergleicht den eingegebenen mit dem erwarteten Zugangscode in konstanter Zeit."""
    return hmac.compare_digest(eingabe.encode(), code.encode())


def waehle_frage() -> str | None:
    """Zeigt Beispielfragen und Eingabefeld; liefert die gestellte Frage oder None."""
    frage = None
    for i, (spalte, beispiel) in enumerate(zip(st.columns(3), BEISPIELFRAGEN)):
        if spalte.button(beispiel, key=f"beispiel_{i}"):
            frage = beispiel
    freitext = st.text_input("Eigene Frage", key="freitext")
    if st.button("Frage stellen", key="frage_stellen") and freitext.strip():
        frage = freitext.strip()
    return frage


def zeige_antwort(antwort: rag.Antwort) -> None:
    """Zeigt Antwort, Warnungen, Quellen und Kennzahlen; bei Ablehnung ohne Quellenliste."""
    ist_ablehnung = antwort.antwort.strip() == ABLEHNUNG
    for warnung in antwort.warnungen:
        st.warning(warnung)
    st.markdown(antwort.antwort)
    if not ist_ablehnung:
        st.markdown("**Quellen**")
        gesehen = set()
        for q in antwort.quellen:
            if (q['typ'], q['nummer']) in gesehen:
                continue
            gesehen.add((q['typ'], q['nummer']))
            st.markdown(f"- {q['typ']} {q['nummer']} - {q['titel']}")
    st.caption(f"Tokens: {antwort.tokens['prompt']} ein / {antwort.tokens['completion']} aus | Latenz: {antwort.latenz['gesamt_ms']:.0f} ms")


def main() -> None:
    """Baut die Seite auf und beantwortet höchstens MAX_FRAGEN Fragen pro Sitzung."""
    load_dotenv()
    st.title("AI-Act Conformity Copilot")
    st.info(DISCLAIMER)

    code = os.getenv("APP_ZUGANGSCODE")
    if not code:
        st.error("APP_ZUGANGSCODE ist nicht gesetzt – die App startet nicht (siehe .env.example).")
        st.stop()
    if not zugang_erlaubt(st.text_input("Zugangscode", type="password", key="zugangscode"), code):
        st.stop()

    frage = waehle_frage()
    if frage is None:
        return
    anzahl = st.session_state.get("fragen_anzahl", 0)
    if anzahl >= MAX_FRAGEN:
        st.warning(f"Pro Sitzung sind höchstens {MAX_FRAGEN} Fragen möglich.")
        return
    st.session_state["fragen_anzahl"] = anzahl + 1
    try:
        with st.spinner("Suche Auszüge und formuliere Antwort …"):
            antwort = rag.beantworte(frage)
    except Exception as exc:  # bewusst breit: jeder Fehler wird nur bereinigt angezeigt
        st.error(f"Fehler {type(exc).__name__}: {bereinige(str(exc))[:MAX_FEHLERLAENGE]}")
        return
    zeige_antwort(antwort)


main()

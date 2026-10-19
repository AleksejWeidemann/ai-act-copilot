# AI-Act Conformity Copilot

Klassifiziert ein KI-System anhand seiner Dokumentation nach EU AI Act und erstellt eine Gap-Analyse gegen Anhang IV, bei der jede Aussage eine Belegstelle nennt.

> **Keine Rechtsberatung – Entscheidungsunterstützung mit menschlicher Prüfung.**

## Lokal starten

Voraussetzung: Python 3.13 und eine `.env` nach Vorlage von `.env.example`.

```
pip install -r requirements-dev.txt
pytest -q
streamlit run src/app.py
```

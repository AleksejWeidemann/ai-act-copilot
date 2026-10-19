"""Tests für src/app.py – laufen mit streamlit.testing (AppTest) und einem Fake für beantworte."""

from pathlib import Path

import dotenv
import pytest
from streamlit.testing.v1 import AppTest

import rag

APP = str(Path(__file__).resolve().parent.parent / "src" / "app.py")
CODE = "richtiger-code"
QUELLEN = [{"typ": "artikel", "nummer": "5", "titel": "Verbotene Praktiken im KI-Bereich", "chunk_nr": 1}]


def antwort(
    text: str, warnungen: list[str] | None = None, quellen: list[dict] | None = None
) -> rag.Antwort:
    """Baut ein festes Antwort-Objekt wie von rag.beantworte()."""
    return rag.Antwort(
        antwort=text,
        quellen=quellen or QUELLEN,
        tokens={"embedding": 9, "prompt": 500, "completion": 40},
        latenz={"embedding_ms": 1.0, "suche_ms": 2.0, "chat_ms": 3.0, "gesamt_ms": 6.0},
        warnungen=warnungen or [],
    )


class FakeBeantworte:
    """Ersetzt rag.beantworte: merkt sich Fragen und liefert eine feste Antwort oder einen Fehler."""

    def __init__(self) -> None:
        self.fragen: list[str] = []
        self.ergebnis: rag.Antwort | Exception = antwort("Nein, verboten nach Art. 5.")

    def __call__(self, frage: str) -> rag.Antwort:
        self.fragen.append(frage)
        if isinstance(self.ergebnis, Exception):
            raise self.ergebnis
        return self.ergebnis


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeBeantworte:
    """Setzt den Zugangscode, verhindert das Lesen der .env und ersetzt rag.beantworte."""
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("APP_ZUGANGSCODE", CODE)
    fake = FakeBeantworte()
    monkeypatch.setattr(rag, "beantworte", fake)
    return fake


def starte(code: str | None = CODE) -> AppTest:
    """Startet die App und gibt optional den Zugangscode ein."""
    at = AppTest.from_file(APP, default_timeout=30).run()
    if code is not None and at.text_input:
        at.text_input(key="zugangscode").input(code).run()
    return at


def alle_texte(at: AppTest) -> str:
    """Sammelt alle sichtbaren Texte der App in einem String."""
    teile = [e.value for e in [*at.markdown, *at.caption, *at.text, *at.warning, *at.info, *at.error]]
    teile += [f"{m.label} {m.value}" for m in at.metric]
    teile += [e.value for e in [*at.title, *at.header, *at.subheader]]
    return "\n".join(str(t) for t in teile)


# --- Zugang -------------------------------------------------------------------


def test_titel_und_disclaimer(fake: FakeBeantworte) -> None:
    at = starte(code=None)
    assert at.title[0].value == "AI-Act Conformity Copilot"
    assert "Keine Rechtsberatung" in at.info[0].value


def test_ohne_app_zugangscode_startet_app_nicht(
    fake: FakeBeantworte, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("APP_ZUGANGSCODE")
    at = starte(code=None)
    assert at.error and "APP_ZUGANGSCODE" in at.error[0].value
    assert not at.text_input and not at.button


def test_falscher_zugangscode_erlaubt_keine_frage(fake: FakeBeantworte) -> None:
    at = starte(code="falsch")
    assert not at.button
    assert fake.fragen == []


def test_richtiger_zugangscode_beantwortet_beispielfrage(fake: FakeBeantworte) -> None:
    at = starte()
    at.button(key="beispiel_0").click().run()
    assert fake.fragen == ["Ist Emotionserkennung am Arbeitsplatz erlaubt?"]
    assert "Nein, verboten nach Art. 5." in alle_texte(at)


def test_freie_frage_wird_gestellt(fake: FakeBeantworte) -> None:
    at = starte()
    at.text_input(key="freitext").input("Was ist ein KI-System?")
    at.button(key="frage_stellen").click().run()
    assert fake.fragen == ["Was ist ein KI-System?"]


# --- Antwortanzeige (zeige_antwort, TODO(human)) ------------------------------


def test_antwort_zeigt_quellen_und_kennzahlen(fake: FakeBeantworte) -> None:
    at = starte()
    at.button(key="beispiel_0").click().run()
    texte = alle_texte(at)
    assert "Verbotene Praktiken im KI-Bereich" in texte
    assert "500" in texte  # Prompt-Tokens


def test_doppelte_quellen_werden_nur_einmal_angezeigt(fake: FakeBeantworte) -> None:
    zweiter_chunk = {**QUELLEN[0], "chunk_nr": 2}
    fake.ergebnis = antwort("Nein, verboten (Art. 5).", quellen=[QUELLEN[0], zweiter_chunk])
    at = starte()
    at.button(key="beispiel_0").click().run()
    assert alle_texte(at).count("Verbotene Praktiken im KI-Bereich") == 1


def test_ablehnung_zeigt_keine_quellen(fake: FakeBeantworte) -> None:
    fake.ergebnis = antwort(rag.ABLEHNUNG)
    at = starte()
    at.button(key="beispiel_2").click().run()
    texte = alle_texte(at)
    assert rag.ABLEHNUNG in texte
    assert "Verbotene Praktiken im KI-Bereich" not in texte


def test_warnungen_werden_angezeigt(fake: FakeBeantworte) -> None:
    fake.ergebnis = antwort("Siehe Art. 99.", ["Art. 99 wird zitiert, steht aber nicht in den Quellen"])
    at = starte()
    at.button(key="beispiel_1").click().run()
    assert any("Art. 99 wird zitiert" in w.value for w in at.warning)


# --- Grenzen und Fehler -------------------------------------------------------


def test_nach_20_fragen_hinweis_statt_antwort(fake: FakeBeantworte) -> None:
    at = starte()
    at.session_state["fragen_anzahl"] = 20
    at.button(key="beispiel_0").click().run()
    assert fake.fragen == []
    assert any("20" in w.value for w in at.warning)


def test_fehler_wird_bereinigt_angezeigt(
    fake: FakeBeantworte, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "geheimer-key-123")
    fake.ergebnis = RuntimeError("Chat fehlgeschlagen: 401 mit Key geheimer-key-123")
    at = starte()
    at.button(key="beispiel_0").click().run()
    assert at.error and "Chat fehlgeschlagen" in at.error[0].value
    assert "geheimer-key-123" not in alle_texte(at)

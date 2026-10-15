"""Tests für src/ingest_aiact.py – parsen die eingecheckte EUR-Lex-HTML-Datei."""

import pytest

import ingest_aiact

ROEMISCH = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV"]


@pytest.fixture(scope="module")
def eintraege() -> list[dict]:
    """Parst die Verordnung einmal für alle Tests."""
    return ingest_aiact.parse_aiact(ingest_aiact.HTML_PFAD, ingest_aiact.QUELLEN_PFAD)


def _nummern(eintraege: list[dict], typ: str) -> set[str]:
    return {e["nummer"] for e in eintraege if e["typ"] == typ}


def _eintrag(eintraege: list[dict], typ: str, nummer: str) -> dict:
    [treffer] = [e for e in eintraege if e["typ"] == typ and e["nummer"] == nummer]
    return treffer


def test_artikel_1_bis_113_vorhanden(eintraege: list[dict]) -> None:
    assert {str(n) for n in range(1, 114)} <= _nummern(eintraege, "artikel")


def test_anhaenge_i_bis_xiv_vorhanden(eintraege: list[dict]) -> None:
    assert set(ROEMISCH) <= _nummern(eintraege, "anhang")


def test_kein_text_leer(eintraege: list[dict]) -> None:
    assert all(e["text"].strip() for e in eintraege)


def test_kein_text_enthaelt_marker(eintraege: list[dict]) -> None:
    assert not [e["nummer"] for e in eintraege if "▼" in e["text"] or "►" in e["text"]]


@pytest.mark.parametrize(("typ", "nummer"), [("artikel", "5"), ("anhang", "III"), ("anhang", "IV")])
def test_kerneintraege_laenger_als_500_zeichen(eintraege: list[dict], typ: str, nummer: str) -> None:
    assert len(_eintrag(eintraege, typ, nummer)["text"]) > 500


def test_mindestens_ein_eintrag_durch_omnibus_geaendert(eintraege: list[dict]) -> None:
    assert any(e["geaendert_durch_omnibus"] for e in eintraege)

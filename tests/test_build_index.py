"""Tests für das Chunking in src/build_index.py – laufen ohne Netzwerk."""

import json
from pathlib import Path
from types import SimpleNamespace

import openai
import pytest

import build_index

JSON_PFAD = Path(__file__).resolve().parent.parent / "data" / "aiact" / "aiact.json"
METADATEN = ("typ", "nummer", "titel", "geaendert_durch_omnibus", "fassung", "url")


def _eintrag(text: str) -> dict:
    """Baut einen Beispiel-Eintrag mit festen Metadaten."""
    return {
        "typ": "artikel",
        "nummer": "5",
        "titel": "Verbotene Praktiken im KI-Bereich",
        "text": text,
        "geaendert_durch_omnibus": True,
        "berichtigt": False,
        "fassung": "konsolidierte Fassung 27.07.2026",
        "url": "https://eur-lex.europa.eu/beispiel",
    }


@pytest.fixture(scope="module")
def aiact() -> list[dict]:
    """Lädt die echten AI-Act-Einträge aus data/aiact/aiact.json."""
    return json.loads(JSON_PFAD.read_text(encoding="utf-8"))


def test_kein_chunk_ueber_2000_zeichen(aiact: list[dict]) -> None:
    for eintrag in aiact:
        for chunk in build_index.chunke_eintrag(eintrag):
            assert len(chunk["text"]) <= build_index.MAX_ZEICHEN, (eintrag["nummer"], chunk["chunk_nr"])


def test_zusammengefuegte_chunks_ergeben_originaltext(aiact: list[dict]) -> None:
    for eintrag in aiact:
        chunks = build_index.chunke_eintrag(eintrag)
        assert "".join(c["text"] for c in chunks) == eintrag["text"], eintrag["nummer"]


def test_metadaten_bleiben_erhalten(aiact: list[dict]) -> None:
    for eintrag in aiact:
        chunks = build_index.chunke_eintrag(eintrag)
        assert [c["chunk_nr"] for c in chunks] == list(range(len(chunks)))
        for chunk in chunks:
            for feld in METADATEN:
                assert chunk[feld] == eintrag[feld]


def test_kurzer_eintrag_bleibt_ein_chunk() -> None:
    chunks = build_index.chunke_eintrag(_eintrag("(1) Kurzer Absatz.\n(2) Noch einer."))
    assert len(chunks) == 1
    assert chunks[0]["text"] == "(1) Kurzer Absatz.\n(2) Noch einer."
    assert chunks[0]["chunk_nr"] == 0


def test_langer_eintrag_wird_an_zeilengrenzen_geteilt() -> None:
    zeilen = [f"{buchstabe}) " + "x" * 900 for buchstabe in "abcd"]
    chunks = build_index.chunke_eintrag(_eintrag("\n".join(zeilen)))
    assert len(chunks) == 2
    assert all(c["text"].startswith(("a)", "c)")) for c in chunks)


def test_zeile_ohne_umbruch_wird_vor_absatznummer_geteilt() -> None:
    text = "(1) " + "Satz eins. " * 150 + "(2) " + "Satz zwei. " * 150
    chunks = build_index.chunke_eintrag(_eintrag(text))
    assert "".join(c["text"] for c in chunks) == text
    assert any(c["text"].startswith("(2) ") for c in chunks)


def test_text_ohne_jede_grenze_wird_hart_geteilt() -> None:
    text = "x" * 4500
    chunks = build_index.chunke_eintrag(_eintrag(text))
    assert [len(c["text"]) for c in chunks] == [2000, 2000, 500]


def test_ids_sind_eindeutig(aiact: list[dict]) -> None:
    chunks = [c for e in aiact for c in build_index.chunke_eintrag(e)]
    assert len({c["id"] for c in chunks}) == len(chunks)


def test_embedding_wiederholt_bei_429(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build_index.time, "sleep", lambda _: None)
    aufrufe = []
    antwort = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.0] * 1536)], usage=SimpleNamespace(prompt_tokens=7)
    )

    def create(**_: object) -> SimpleNamespace:
        aufrufe.append(1)
        if len(aufrufe) == 1:
            raise openai.RateLimitError(
                "429", response=SimpleNamespace(request=None, status_code=429, headers={}), body=None
            )
        return antwort

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    vektoren, tokens = build_index.bette_ein_mit_retry(client, "emb", ["Hochrisiko"])
    assert len(aufrufe) == 2
    assert tokens == 7 and len(vektoren[0]) == 1536

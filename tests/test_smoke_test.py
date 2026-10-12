"""Tests für src/smoke_test.py – laufen ohne Netzwerk mit Fake-Clients."""

from types import SimpleNamespace

import pytest

import smoke_test


def _fake_openai_client() -> SimpleNamespace:
    """Baut einen Fake-Client mit chat.completions.create und embeddings.create."""
    chat_antwort = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Verbindung steht."))],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=4),
    )
    embedding_antwort = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1] * 1536)],
        usage=SimpleNamespace(prompt_tokens=5),
    )
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: chat_antwort)),
        embeddings=SimpleNamespace(create=lambda **_: embedding_antwort),
    )


def test_chat_pruefung_nennt_tokens_und_antwort() -> None:
    details = smoke_test.pruefe_chat(_fake_openai_client(), "chat")
    assert "12 ein" in details and "4 aus" in details
    assert "Verbindung steht." in details


def test_embedding_pruefung_nennt_vektorlaenge_und_tokens() -> None:
    details = smoke_test.pruefe_embedding(_fake_openai_client(), "embedding")
    assert "1536" in details and "5" in details


def test_suche_pruefung_zaehlt_indizes() -> None:
    index_client = SimpleNamespace(list_index_names=lambda: iter([]))
    assert "0 Indizes" in smoke_test.pruefe_suche(index_client)


def test_fehler_stoppt_nicht_die_folgenden_pruefungen() -> None:
    def kaputt() -> str:
        raise RuntimeError("Verbindung abgelehnt")

    ergebnisse = smoke_test.fuehre_pruefungen_aus(
        [("Chat", kaputt), ("Embedding", lambda: "ok"), ("Suche", lambda: "ok")]
    )
    assert [e.ok for e in ergebnisse] == [False, True, True]
    assert "Verbindung abgelehnt" in ergebnisse[0].details


def test_fehlermeldung_enthaelt_keine_keys_oder_endpunkte(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "geheimer-key-123")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://meine-ressource.openai.azure.com/")

    def leakt() -> str:
        raise RuntimeError(
            "401 bei meine-ressource.openai.azure.com mit Key geheimer-key-123"
        )

    [ergebnis] = smoke_test.fuehre_pruefungen_aus([("Chat", leakt)])
    zeile = smoke_test.formatiere(ergebnis)
    assert "geheimer-key-123" not in zeile
    assert "meine-ressource" not in zeile
    assert "FEHLER" in zeile


def test_fehlende_umgebungsvariable_gibt_verstaendliche_meldung(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AZURE_OPENAI_CHAT_DEPLOYMENT", raising=False)
    with pytest.raises(RuntimeError, match="AZURE_OPENAI_CHAT_DEPLOYMENT"):
        smoke_test.env("AZURE_OPENAI_CHAT_DEPLOYMENT")

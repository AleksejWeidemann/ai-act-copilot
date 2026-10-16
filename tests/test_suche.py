"""Tests für src/suche.py – laufen ohne Netzwerk mit Fake-Clients."""

from types import SimpleNamespace

import pytest

import suche


class FakeSearchClient:
    """Merkt sich die Suchparameter und liefert feste Treffer."""

    def __init__(self, treffer: list[dict]) -> None:
        self.treffer = treffer
        self.aufrufe: list[dict] = []

    def search(self, **kwargs: object) -> list[dict]:
        self.aufrufe.append(kwargs)
        return self.treffer


def test_einbettung_nutzt_embedding_deployment_und_liefert_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "emb-test")
    aufrufe: list[dict] = []

    def create(**kwargs: object) -> SimpleNamespace:
        aufrufe.append(kwargs)
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2])], usage=SimpleNamespace(prompt_tokens=7)
        )

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    vektor, tokens = suche.bette_frage_ein(client, "Frage")
    assert vektor == [0.1, 0.2]
    assert tokens == 7
    assert aufrufe[0]["model"] == "emb-test"


def test_hybrid_suche_holt_top5_mit_text_und_vektor() -> None:
    client = FakeSearchClient([{"typ": "artikel", "nummer": "5"}])
    treffer = suche.hybrid_suche(client, "Emotionserkennung", [0.1], felder=["typ", "text"])
    assert treffer == [{"typ": "artikel", "nummer": "5"}]
    aufruf = client.aufrufe[0]
    assert aufruf["search_text"] == "Emotionserkennung"
    assert aufruf["top"] == 5
    assert aufruf["select"] == ["typ", "text"]
    [vektor_anfrage] = aufruf["vector_queries"]
    assert vektor_anfrage.k_nearest_neighbors == 5
    assert vektor_anfrage.fields == "vector"

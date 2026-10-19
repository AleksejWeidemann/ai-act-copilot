"""Tests für src/rag.py und src/ask_cli.py – laufen ohne Netzwerk mit Fake-Clients."""

import re
from types import SimpleNamespace

import pytest

import ask_cli
import rag

TREFFER = [
    {
        "typ": "artikel",
        "nummer": "5",
        "titel": "Verbotene Praktiken im KI-Bereich",
        "chunk_nr": 1,
        "text": "f) das Inverkehrbringen ... zur Ableitung von Emotionen am Arbeitsplatz ...",
        "fassung": "konsolidierte Fassung 27.07.2026",
        "@search.score": 0.03,
    },
    {
        "typ": "anhang",
        "nummer": "III",
        "titel": "Hochrisiko-KI-Systeme gemäß Artikel 6 Absatz 2",
        "chunk_nr": 0,
        "text": "1. Biometrie ... c) KI-Systeme zur Emotionserkennung.",
        "fassung": "konsolidierte Fassung 27.07.2026",
        "@search.score": 0.02,
    },
]


class FakeSearchClient:
    """Liefert feste Treffer und merkt sich die Suchparameter."""

    def __init__(self) -> None:
        self.aufrufe: list[dict] = []

    def search(self, **kwargs: object) -> list[dict]:
        self.aufrufe.append(kwargs)
        return TREFFER


class FakeOpenAI:
    """Fake für embeddings.create und chat.completions.create."""

    def __init__(self, antworttext: str = "Nein, verboten nach Art. 5.") -> None:
        self.chat_aufrufe: list[dict] = []
        self.antworttext = antworttext
        self.embeddings = SimpleNamespace(create=self._embedding)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))

    def _embedding(self, **_: object) -> SimpleNamespace:
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1] * 3)], usage=SimpleNamespace(prompt_tokens=9)
        )

    def _chat(self, **kwargs: object) -> SimpleNamespace:
        self.chat_aufrufe.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.antworttext))],
            usage=SimpleNamespace(prompt_tokens=500, completion_tokens=40),
        )


@pytest.fixture
def kette(monkeypatch: pytest.MonkeyPatch) -> FakeOpenAI:
    """Setzt Deployments und ersetzt die TODO(human)-Teile, damit die Kette allein testbar ist."""
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "emb-test")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "chat-test")
    monkeypatch.setattr(rag, "formatiere_auszug", lambda t: f"[{t['nummer']}] {t['text']}")
    monkeypatch.setattr(rag, "pruefe_zitate", lambda antwort, quellen: [])
    return FakeOpenAI()


# --- Kette (Infrastruktur) ---------------------------------------------------


def test_chat_wird_mit_deployment_und_tokenlimit_aufgerufen(kette: FakeOpenAI) -> None:
    rag.beantworte("Ist Emotionserkennung erlaubt?", kette, FakeSearchClient())
    [aufruf] = kette.chat_aufrufe
    assert aufruf["model"] == "chat-test"
    assert aufruf["max_completion_tokens"] == 800


def test_prompt_enthaelt_regeln_auszuege_und_frage(kette: FakeOpenAI) -> None:
    rag.beantworte("Ist Emotionserkennung erlaubt?", kette, FakeSearchClient())
    system, user = kette.chat_aufrufe[0]["messages"]
    assert system == {"role": "system", "content": rag.SYSTEM_PROMPT}
    assert "Ableitung von Emotionen am Arbeitsplatz" in user["content"]
    assert "Biometrie" in user["content"]
    assert "Ist Emotionserkennung erlaubt?" in user["content"]


def test_ablehnungssatz_steht_woertlich_im_system_prompt() -> None:
    assert rag.ABLEHNUNG == "Dazu enthalten die Auszüge keine Angaben."
    assert f"antworte genau mit dem Satz: {rag.ABLEHNUNG}\n" in rag.SYSTEM_PROMPT


def test_suche_fordert_text_der_chunks_an(kette: FakeOpenAI) -> None:
    search_client = FakeSearchClient()
    rag.beantworte("Frage", kette, search_client)
    assert "text" in search_client.aufrufe[0]["select"]


def test_ergebnis_enthaelt_antwort_quellen_tokens_latenz(kette: FakeOpenAI) -> None:
    ergebnis = rag.beantworte("Frage", kette, FakeSearchClient())
    assert ergebnis.antwort == "Nein, verboten nach Art. 5."
    assert ergebnis.quellen == [
        {"typ": "artikel", "nummer": "5", "titel": "Verbotene Praktiken im KI-Bereich", "chunk_nr": 1},
        {
            "typ": "anhang",
            "nummer": "III",
            "titel": "Hochrisiko-KI-Systeme gemäß Artikel 6 Absatz 2",
            "chunk_nr": 0,
        },
    ]
    assert ergebnis.tokens == {"embedding": 9, "prompt": 500, "completion": 40}
    assert set(ergebnis.latenz) == {"embedding_ms", "suche_ms", "chat_ms", "gesamt_ms"}
    assert all(wert >= 0 for wert in ergebnis.latenz.values())


def test_fehler_im_modellaufruf_nennt_schritt_ohne_geheimnisse(
    kette: FakeOpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "geheimer-key-123")

    def kaputt(**_: object) -> None:
        raise ConnectionError("401 mit Key geheimer-key-123")

    kette.chat.completions.create = kaputt
    with pytest.raises(RuntimeError, match="Chat") as info:
        rag.beantworte("Frage", kette, FakeSearchClient())
    assert "geheimer-key-123" not in str(info.value)


def test_cli_zeigt_warnungen_und_disclaimer() -> None:
    ergebnis = rag.Antwort(
        antwort="Siehe Art. 99.",
        quellen=[{"typ": "artikel", "nummer": "5", "titel": "Verbote", "chunk_nr": 0}],
        tokens={"embedding": 1, "prompt": 2, "completion": 3},
        latenz={"embedding_ms": 1.0, "suche_ms": 2.0, "chat_ms": 3.0, "gesamt_ms": 6.0},
        warnungen=["Artikel 99 ist nicht unter den gefundenen Quellen."],
    )
    ausgabe = ask_cli.formatiere_ergebnis(ergebnis)
    assert "WARNUNG" in ausgabe and "Artikel 99" in ausgabe
    assert "Keine Rechtsberatung" in ausgabe


# --- TODO(human)-Teile: werden grün, sobald (a), (b), (c) umgesetzt sind -------


def test_a_system_prompt_ist_gesetzt() -> None:
    assert rag.SYSTEM_PROMPT.strip(), "TODO(human) (a): SYSTEM_PROMPT ist noch leer"


def test_b_auszug_nennt_artikel_und_text() -> None:
    auszug = rag.formatiere_auszug(TREFFER[0])
    assert re.search(r"Art(ikel|\.)\s*5\b", auszug)
    assert "Ableitung von Emotionen am Arbeitsplatz" in auszug


def test_b_auszug_nennt_anhang() -> None:
    auszug = rag.formatiere_auszug(TREFFER[1])
    assert "Anhang III" in auszug
    assert "Emotionserkennung" in auszug


QUELLEN = [
    {"typ": "artikel", "nummer": "5", "titel": "Verbote", "chunk_nr": 1},
    {"typ": "artikel", "nummer": "4a", "titel": "…", "chunk_nr": 0},
    {"typ": "anhang", "nummer": "III", "titel": "Hochrisiko", "chunk_nr": 0},
]


@pytest.mark.parametrize(
    "antwort",
    [
        "Das ist verboten (Art. 5 Abs. 1 Buchst. f).",
        "Das ist verboten (Artikel 5) und hochriskant (Anhang III).",
        "Es gilt eine Sonderregel (Art. 4a).",
        "Dazu enthalten die Auszüge keine Angaben.",
    ],
)
def test_c_belegte_zitate_ergeben_keine_warnung(antwort: str) -> None:
    assert rag.pruefe_zitate(antwort, QUELLEN) == []


@pytest.mark.parametrize(
    ("antwort", "quellen"),
    [
        (
            "Wenn ein Risiko nach Artikel 79 Absatz 1 vermutet wird (Art. 26 Abs. 5).",
            [{"typ": "artikel", "nummer": "26", "titel": "Pflichten der Betreiber", "chunk_nr": 2}],
        ),
        (
            "Siehe Anhang I Abschnitt A (Art. 11 Abs. 2).",
            [{"typ": "artikel", "nummer": "11", "titel": "Technische Dokumentation", "chunk_nr": 0}],
        ),
    ],
)
def test_c_querverweise_im_fliesstext_werden_nicht_geprueft(
    antwort: str, quellen: list[dict]
) -> None:
    assert rag.pruefe_zitate(antwort, quellen) == []


@pytest.mark.parametrize(
    ("antwort", "kennung"),
    [
        ("Das ist geregelt (Artikel 99).", "99"),
        ("Es besteht eine Pflicht (Art. 50 Abs. 3).", "50"),
        ("Das muss enthalten sein (Anhang IV).", "IV"),
        ("Das gilt (Art 99).", "99"),
        ("Das gilt (Art. 99) und nochmals (Art. 99).", "99"),
        ("(Art. 79)", "79"),
    ],
)
def test_c_unbelegtes_zitat_ergibt_warnung(antwort: str, kennung: str) -> None:
    warnungen = rag.pruefe_zitate(antwort, QUELLEN)
    assert len(warnungen) == 1
    assert kennung in warnungen[0]

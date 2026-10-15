"""Parst die konsolidierte EUR-Lex-Fassung des AI Act in data/aiact/aiact.json.

Die Struktur kommt aus dem HTML: Artikel sind <div id="art_N">, Anhänge
<div id="anx_N">. Änderungsmarker (▼B, ▼M1, ▼C1, ▼C2) stehen als
<p class="modref"> bzw. <p class="arrow">. Ein Marker gilt für den Text
danach bis zum nächsten Marker – er kann daher auch vor einem Eintrag
stehen (z. B. ▼M1 vor dem eingefügten Artikel 4a).
"""

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag

DATEN_ORDNER = Path(__file__).resolve().parent.parent / "data" / "aiact"
HTML_PFAD = DATEN_ORDNER / "ai_act_de.html"
QUELLEN_PFAD = DATEN_ORDNER / "QUELLEN.md"
JSON_PFAD = DATEN_ORDNER / "aiact.json"

FASSUNG = "konsolidierte Fassung 27.07.2026"
EINTRAG_ID = re.compile(r"^(art|anx)_([^.]+)$")
MARKER_KLASSEN = ["modref", "arrow"]
ABSATZ_ENDE = "\x00"


def lese_quell_url(quellen_pfad: Path) -> str:
    """Liest die URL der HTML-Datei aus QUELLEN.md (Zeile 'ai_act_de.html: <url>')."""
    for zeile in quellen_pfad.read_text(encoding="utf-8").splitlines():
        treffer = re.search(r"ai_act_de\.html:\s*(\S+)", zeile)
        if treffer:
            return treffer.group(1)
    raise ValueError(f"Keine URL für ai_act_de.html in {quellen_pfad}")


def marker_name(marker: Tag) -> str:
    """Liefert den Kürzel eines Markers, z. B. 'M1' für '▼M1 —————'."""
    treffer = re.search(r"[▼►]\s*([A-Z]+\d*)", marker.get_text())
    return treffer.group(1) if treffer else ""


def wirksame_marker(eintrag: Tag, alle_marker: list[Tag], position: dict[int, int]) -> set[str]:
    """Sammelt die Marker im Eintrag plus den Marker, der zu Beginn des Eintrags gilt."""
    start = position[id(eintrag)]
    innen = {marker_name(m) for m in eintrag.find_all(class_=MARKER_KLASSEN)}
    davor = [m for m in alle_marker if position[id(m)] < start]
    if davor:
        innen.add(marker_name(davor[-1]))
    return innen


def titel_element(eintrag: Tag, typ: str) -> Tag | None:
    """Findet das Titel-Element; Anhang XIV hat keine title-annex-2, dort steht der Titel im ersten Absatz."""
    if typ == "artikel":
        return eintrag.find(class_="stitle-article-norm")
    titel = eintrag.find(class_="title-annex-2")
    if titel is None:
        kopf = eintrag.find(class_="title-annex-1")
        titel = kopf.find_next_sibling("p", class_="norm") if kopf else None
    return titel


def bereinige(text: str) -> str:
    """Normalisiert Leerzeichen und entfernt leere Zeilen."""
    zeilen = (re.sub(r"\s+", " ", z.replace("\xa0", " ")).strip() for z in text.splitlines())
    return "\n".join(z for z in zeilen if z)


def eintrag_text(eintrag: Tag, typ: str) -> str:
    """Extrahiert den Fließtext ohne Marker, Überschrift und Titel; ein Absatz je Zeile."""
    kopie = BeautifulSoup(str(eintrag), "html.parser").find("div")
    titel = titel_element(kopie, typ)
    if titel is not None:
        titel.decompose()
    for element in kopie.find_all(class_=MARKER_KLASSEN + ["title-article-norm", "title-annex-1"]):
        element.decompose()
    for absatz in kopie.find_all(["p", "tr"]):
        absatz.append(ABSATZ_ENDE)
    # Zeilenumbrüche aus dem HTML-Quelltext sind keine Absätze, nur unsere Absatzenden zählen.
    return bereinige(kopie.get_text().replace("\n", " ").replace(ABSATZ_ENDE, "\n"))


def parse_eintrag(eintrag: Tag, marker: set[str], url: str) -> dict:
    """Baut den JSON-Eintrag für einen Artikel oder Anhang."""
    kuerzel, nummer = EINTRAG_ID.match(eintrag["id"]).groups()
    typ = "artikel" if kuerzel == "art" else "anhang"
    titel = titel_element(eintrag, typ)
    return {
        "typ": typ,
        "nummer": nummer,
        "titel": bereinige(titel.get_text()) if titel else "",
        "text": eintrag_text(eintrag, typ),
        "geaendert_durch_omnibus": "M1" in marker,
        "berichtigt": bool(marker & {"C1", "C2"}),
        "fassung": FASSUNG,
        "url": url,
    }


def parse_aiact(html_pfad: Path, quellen_pfad: Path) -> list[dict]:
    """Parst alle Artikel und Anhänge in Dokumentreihenfolge."""
    soup = BeautifulSoup(html_pfad.read_text(encoding="utf-8"), "html.parser")
    position = {id(tag): i for i, tag in enumerate(soup.find_all(True))}
    alle_marker = soup.find_all(class_=MARKER_KLASSEN)
    url = lese_quell_url(quellen_pfad)
    return [
        parse_eintrag(e, wirksame_marker(e, alle_marker, position), url)
        for e in soup.find_all("div", id=EINTRAG_ID)
    ]


def main() -> None:
    """Schreibt aiact.json und gibt eine Zusammenfassung aus."""
    eintraege = parse_aiact(HTML_PFAD, QUELLEN_PFAD)
    JSON_PFAD.write_text(
        json.dumps(eintraege, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    artikel = sum(e["typ"] == "artikel" for e in eintraege)
    print(f"Artikel: {artikel}")
    print(f"Anhänge: {len(eintraege) - artikel}")
    print(f"Durch Omnibus (M1) geändert: {sum(e['geaendert_durch_omnibus'] for e in eintraege)}")
    print(f"Berichtigt (C1/C2): {sum(e['berichtigt'] for e in eintraege)}")
    print(f"Geschrieben: {JSON_PFAD}")


if __name__ == "__main__":
    main()

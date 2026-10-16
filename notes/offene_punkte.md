# Offene Punkte

- [ ] RAG-Aufruf: max_completion_tokens explizit setzen (Kostenbremse gegen lange Antworten)
- [ ] README: Scope-Entscheidungen notieren (preview-API-Version, Free-Tier ohne SLA, Lock-Datei bewusst weggelassen)
- [ ] V2: Omnibus-Markierung pro Absatz statt pro Eintrag
- [ ] Test mit festem Mini-HTML-Beispiel für die Marker-Zuordnung
- [ ] Test, dass aiact.json dem aktuellen Parser-Ergebnis entspricht
- [ ] Retrieval-Vielfalt: höchstens 2 Chunks pro Artikel in den Top-k
- [ ] Auszüge um Absatz-Kontext ergänzen (z. B. "Art. 5 Abs. 1, Fortsetzung"),
      damit genaue Zitate möglich sind; Absatznummer in die Chunk-Metadaten
- [ ] Ablehnungssatz als Konstante ABLEHNUNG (für Golden Set)
- [ ] Zitat-Prüfung V2: Aufzählungen ("Art. 5 und 6", "Anhänge III und IV"),
      Artikel anderer Gesetze ("Art. 6 DSGVO")
- [ ] Bekannte Lücken der Zitat-Prüfung als xfail-Tests dokumentieren
- [ ] Golden Set: Inhaltskriterien statt Wortlaut prüfen (Ausnahme genannt?,
      Artikel zitiert?), mehrere Läufe pro Frage
- [ ] Bei Ablehnung keine Quellenliste anzeigen (UI)
- [ ] Test, dass Fehlermeldungen in ask_cli bereinigt werden
- [ ] bereinige() in eigenes Modul src/sicherheit.py verschieben

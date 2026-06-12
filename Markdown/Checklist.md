# Checklist

## Ideen aus Idea.md übernehmen

- [ ] Neue Themen aus [Idea.md](Idea.md) eintragen
- [ ] Relevante Punkte als implementiert / offen markieren
- [ ] Neue Modelle, Strategien oder Frameworks in die passenden Abschnitte einsortieren

## Aktueller Stand

- [x] Ziel des Projekts festgelegt: AutoML für Anomalieerkennung auf dem TEP-Datensatz
- [x] Fokus auf modularen Aufbau mit austauschbaren Bausteinen definiert
- [x] Erste Version auf unsupervised learning ausgerichtet
- [x] Batch-Verarbeitung als Startpunkt festgelegt
- [x] Parquet-Dateien nach `automl/data/` verschoben
- [x] TEP-Splits als Ladefunktion implementiert
- [x] Ein erster Minimal-Workflow für unsupervised learning läuft
- [x] Registry für austauschbare Detektoren eingebaut
- [x] Vergleichsworkflow für mehrere Detektoren eingebaut
- [x] README um Konzept und Workflow ergänzt

## Implementiert

- [x] Isolation Forest als erste Baseline implementiert
- [x] Local Outlier Factor als zweites Vergleichsmodell implementiert
- [x] TEP-Daten werden als vier Splits geladen: Training / Testing, fault-free / faulty
- [x] Feasible unsupervised training auf `train_fault_free`
- [x] Evaluation auf kombiniertem Testsplit mit Labels
- [x] Metriken: PR-AUC, ROC-AUC und F1
- [x] Laufzeitmessung im Evaluationslauf
- [x] Registry-basierte Modellauswahl
- [x] Default-Workflow bleibt schnell und nutzt nur einen Detektor
- [x] Vergleichsworkflow kann mehrere Detektoren gezielt ausführen

## Noch nicht implementiert

- [ ] PyOD als Referenz ganz am Ende integrieren
- [ ] Einen dritten Detektor ergänzen
- [ ] Suchstrategien parametrisieren statt nur Detektoren zu wechseln
- [ ] SMAC als Suchstrategie implementieren
- [ ] Irace als Suchstrategie implementieren
- [ ] Meta-Learning als Initialisierung einsetzen
- [ ] Semi-supervised learning vorbereiten
- [ ] Streaming / Online-Detection vorbereiten
- [ ] Ergebnisse systematisch speichern
- [ ] Ein kleines CLI oder Skript für wiederholbare Runs bauen

## Forschungsstand, noch offen in der Implementierung

- [x] ParamILS als mögliche Suchstrategie bekannt
- [x] SMAC als mögliche Suchstrategie bekannt
- [x] GGA als möglicher Suchansatz bekannt
- [x] Irace als mögliche Suchstrategie bekannt
- [x] Meta-Learning als möglicher Ansatz bekannt
- [x] AnoGAN als mögliche neuronale Methode bekannt
- [x] PyOD als Referenzframework bekannt
- [x] LSCP als Vergleichsframework bekannt
- [x] TODS als Vergleichsframework bekannt
- [x] AutoOD als Vergleichsframework bekannt

## Nächste Arbeitspakete

- [ ] One-Class SVM als weitere Baseline prüfen
- [ ] Autoencoder als neuronale Baseline prüfen
- [ ] AnoGAN als späteres Deep-Learning-Modell prüfen
- [ ] Vergleichstabelle für Methoden und Frameworks erstellen
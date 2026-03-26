````markdown
# TikTok Virality Predictor

Dieses Projekt implementiert eine Machine-Learning-Pipeline, die vorhersagt, ob ein TikTok-Video "viral" geht (definiert als das Erreichen der Top 20% der Aufrufzahlen). Die Klassifikation basiert ausschließlich auf Video-Metadaten und den zugehörigen Transkriptionstexten.

## Übersicht und Ergebnisse

Das Ziel des Projekts war der Aufbau einer vollständigen Data-Science-Pipeline von der Datenbereinigung über Feature Engineering bis hin zur Evaluierung. Eine zentrale Herausforderung war der stark unbalancierte Datensatz (80/20 Split).

* **Bestes Modell:** Logistic Regression
* **Recall (Viral Class):** 1.00 (100%)
* **F1-Score:** 0.576
* **Angewandte Methoden:** TF-IDF (inkl. N-Grams), Probability Threshold Tuning, Class Weighting.

Hinweis zur Modellperformance: Ein F1-Score von ca. 0.58 stellt für diesen Datensatz ein realistisches Limit dar. Faktoren wie Veröffentlichungszeitpunkt oder der Zufallsfaktor des Algorithmus lassen sich aus Pre-Upload-Metadaten nicht ableiten. Um das Verpassen potenziell viraler Trends zu minimieren, wurde das Modell durch Threshold-Anpassung gezielt auf einen maximalen Recall kalibriert.

## Technologien

* Python 3
* pandas
* numpy
* scikit-learn
* xgboost

## Feature Engineering

Um Data Leakage zu vermeiden, wurden alle Post-Upload-Metriken (Likes, Shares, Comments) konsequent aus dem Trainingsset entfernt. Aus den verbleibenden Daten wurden folgende Features generiert:

1. **Textmetriken:** Wortanzahl, Textlänge, Anteil an Großbuchstaben sowie die Häufigkeit von Satz- und Fragezeichen.
2. **Hook-Extraktion:** Die ersten drei Wörter des Videos wurden als separates kategorisches Merkmal extrahiert.
3. **NLP:** Der Rohtext wurde mittels `TfidfVectorizer` in numerische Vektoren transformiert (unter Ausschluss von Stopwörtern und Einbezug von Bi- und Tri-Grams).

## Projektstruktur

* `dataloader.py`: Laden der Rohdaten.
* `preprocess.py`: Datenbereinigung, Label-Generierung (Target) und Feature Engineering.
* `model.py`: Definition der Scikit-Learn Pipelines, Modeltraining, Threshold Tuning und Metrik-Evaluierung.
* `main.py`: Einstiegspunkt zur Ausführung der gesamten Pipeline.
* `data/tiktok_dataset.csv`: Der verwendete Datensatz.

## Installation und Ausführung

Repository klonen:
```bash
git clone [https://github.com/USERNAME/tiktok-virality-predictor.git](https://github.com/USERNAME/tiktok-virality-predictor.git)
cd tiktok-virality-predictor
```

Abhängigkeiten installieren:
```bash
pip install pandas numpy scikit-learn xgboost
```

Pipeline starten:
```bash
python main.py
```
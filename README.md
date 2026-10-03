# Supervised WSD for Spoken News

## Overview

News transcripts are full of words like bank, interest, plant and run whose
meaning depends on context. This project trains a linear SVM and Naive Bayes
per word to predict the WordNet sense from POS tags and a bag-of-words over a
±2 or ±5 window, using SemCor as training data and 5-fold cross-validation.

## Running

1. Create and activate a virtual environment.
   ```
   python -m venv .venv && source .venv/bin/activate
   ```

2. Install the dependencies.
   ```
   pip install -r requirements.txt
   ```

3. Download the NLTK data (SemCor, WordNet, stopwords and the POS tagger).
   ```
   python -m nltk.downloader semcor wordnet stopwords averaged_perceptron_tagger_eng
   ```

4. Build the dataset from SemCor into `data/annotated/semcor_wsd.csv`.
   ```
   python src/preprocessing.py
   ```

5. Train and test both models at both window sizes. This prints a summary and
   writes `results/metrics.csv` and `results/predictions.csv`.
   ```
   python src/pipeline.py
   ```

6. (Optional) Rebuild the chart used in the evaluation report.
   ```
   python docs/figures.py
   ```

## Results

| | Accuracy (±2) | Accuracy (±5) | Macro F1 (±2) | Macro F1 (±5) |
|---|---|---|---|---|
| SVM | 0.65 | 0.64 | 0.54 | 0.50 |
| Naive Bayes | 0.61 | 0.61 | 0.40 | 0.41 |

The SVM does better, mostly on rare senses. The wider window only helps words
whose senses depend on topic, like bank. Per-word numbers and failure analysis
are in `docs/Evaluation_Report.pdf`.

## Layout

```
src/
  preprocessing.py   dataset extraction from SemCor, POS tags, lemmas, stopwords
  models.py          feature extraction, SVM and Naive Bayes
  pipeline.py        cross-validated training and evaluation
data/annotated/      the extracted SemCor dataset
docs/                methodology and evaluation report (LaTeX + PDF)
```

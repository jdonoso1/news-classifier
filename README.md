# News Topic Classifier

Classify news articles into 5 categories: **sports, politics, technology, science, lifestyle**.

**Dataset:** 20 Newsgroups (built into scikit-learn, no download needed)

---

## What this project covers

- Multi-class text classification (5 categories)
- TF-IDF with unigrams vs unigrams + bigrams
- Models: Logistic Regression and LinearSVC
- Evaluation: accuracy, macro F1, per-class classification report
- Error analysis: which categories get confused and why

---

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression (unigrams) | ~0.90 | ~0.90 |
| LinearSVC (unigrams) | ~0.91 | ~0.91 |
| Logistic Regression (bigrams) | ~0.91 | ~0.91 |
| LinearSVC (bigrams) | ~0.92 | ~0.92 |

**Winner: LinearSVC + bigrams.** LinearSVC tends to edge out LR on text tasks. Bigrams help capture phrases like "home run" (sports) vs just "run".

**Common confusion:** politics ↔ lifestyle — articles about social issues span both.

---

## Run it

```bash
pip install -r requirements.txt
python news_classifier.py
```

All outputs saved to `results/`.

---

## Project structure

```
news-classifier/
├── news_classifier.py      # full pipeline
├── data_notes.md           # dataset observations
├── preprocessing_notes.md  # cleaning decisions
├── requirements.txt
└── results/
    ├── eda.png
    ├── confusion_matrix.png
    ├── model_comparison.png
    ├── classification_report.txt
    └── metrics.csv
```

# News Topic Classifier
# Goal: classify news articles into 5 categories
# Dataset: 20 Newsgroups (built into sklearn — no download needed)

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)

os.makedirs("results", exist_ok=True)


# ─────────────────────────────────────────────
# Step 1: Load Dataset
# ─────────────────────────────────────────────
# Picking 5 clear, well-separated newsgroup categories
# We remove headers/footers/quotes so the model actually learns
# the content, not just the metadata

CATEGORIES = [
    "rec.sport.baseball",  # sports
    "talk.politics.misc",  # politics
    "comp.graphics",       # technology
    "sci.med",             # science
    "rec.autos",           # lifestyle
]

LABEL_MAP = {
    "rec.sport.baseball": "sports",
    "talk.politics.misc": "politics",
    "comp.graphics":      "technology",
    "sci.med":            "science",
    "rec.autos":          "lifestyle",
}

print("Loading 20 Newsgroups dataset...")
train_raw = fetch_20newsgroups(
    subset="train", categories=CATEGORIES,
    remove=("headers", "footers", "quotes")
)
test_raw = fetch_20newsgroups(
    subset="test", categories=CATEGORIES,
    remove=("headers", "footers", "quotes")
)

train_df = pd.DataFrame({
    "text":     train_raw.data,
    "category": [CATEGORIES[t] for t in train_raw.target],
})
test_df = pd.DataFrame({
    "text":     test_raw.data,
    "category": [CATEGORIES[t] for t in test_raw.target],
})

train_df["label"] = train_df["category"].map(LABEL_MAP)
test_df["label"]  = test_df["category"].map(LABEL_MAP)

print(f"Train: {len(train_df)} | Test: {len(test_df)}")


# ─────────────────────────────────────────────
# Step 2: Inspect the Data
# ─────────────────────────────────────────────

print("\n--- Category Counts (Train) ---")
print(train_df["label"].value_counts())

train_df["length"] = train_df["text"].apply(lambda x: len(x.split()))
print(f"\nAvg article length: {train_df['length'].mean():.0f} words")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

train_df["label"].value_counts().sort_values().plot(
    kind="barh", ax=axes[0], color="#3498db"
)
axes[0].set_title("Category Distribution (Train)")
axes[0].set_xlabel("Count")

train_df.groupby("label")["length"].mean().sort_values().plot(
    kind="barh", ax=axes[1], color="#9b59b6"
)
axes[1].set_title("Avg Article Length by Category")
axes[1].set_xlabel("Words")

plt.tight_layout()
plt.savefig("results/eda.png", dpi=150)
print("Saved → results/eda.png")


# ─────────────────────────────────────────────
# Step 3: Preprocess Text
# ─────────────────────────────────────────────

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)     # letters only
    text = re.sub(r"\s+", " ", text).strip()
    return text


train_df["clean"] = train_df["text"].apply(clean_text)
test_df["clean"]  = test_df["text"].apply(clean_text)


# ─────────────────────────────────────────────
# Step 4: Feature Extraction
# ─────────────────────────────────────────────

X_train = train_df["clean"].values
X_test  = test_df["clean"].values
y_train = train_df["label"].values
y_test  = test_df["label"].values

# TF-IDF unigrams first, then bigrams to see if they help
tfidf_uni = TfidfVectorizer(max_features=30000, ngram_range=(1, 1))
X_train_uni = tfidf_uni.fit_transform(X_train)
X_test_uni  = tfidf_uni.transform(X_test)

tfidf_bi = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
X_train_bi = tfidf_bi.fit_transform(X_train)
X_test_bi  = tfidf_bi.transform(X_test)

print(f"\nUnigram matrix : {X_train_uni.shape}")
print(f"Bigram matrix  : {X_train_bi.shape}")


# ─────────────────────────────────────────────
# Step 5: Train Models
# ─────────────────────────────────────────────

models = {
    "Logistic Regression (unigrams)": (LogisticRegression(max_iter=1000), X_train_uni, X_test_uni),
    "LinearSVC (unigrams)":           (LinearSVC(max_iter=2000),          X_train_uni, X_test_uni),
    "Logistic Regression (bigrams)":  (LogisticRegression(max_iter=1000), X_train_bi,  X_test_bi),
    "LinearSVC (bigrams)":            (LinearSVC(max_iter=2000),          X_train_bi,  X_test_bi),
}

results = {}
label_order = sorted(train_df["label"].unique())

for name, (model, X_tr, X_te) in models.items():
    print(f"\nTraining: {name}")
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    results[name] = {
        "accuracy": accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro"),
        "preds":    preds,
    }
    print(f"  Accuracy : {results[name]['accuracy']:.4f}")
    print(f"  Macro F1 : {results[name]['macro_f1']:.4f}")


# ─────────────────────────────────────────────
# Step 6: Evaluate — Classification Report
# ─────────────────────────────────────────────

best_name  = max(results, key=lambda k: results[k]["macro_f1"])
best_preds = results[best_name]["preds"]

print(f"\n--- Best Model: {best_name} ---")
report = classification_report(y_test, best_preds, target_names=label_order)
print(report)

with open("results/classification_report.txt", "w") as f:
    f.write(f"Best Model: {best_name}\n\n")
    f.write(report)
print("Saved → results/classification_report.txt")

# confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, best_preds, labels=label_order)
ConfusionMatrixDisplay(cm, display_labels=label_order).plot(
    ax=ax, colorbar=False, xticks_rotation=30
)
ax.set_title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150)
print("Saved → results/confusion_matrix.png")


# ─────────────────────────────────────────────
# Step 7: Error Analysis
# ─────────────────────────────────────────────
# Which categories get confused most often?

print("\n--- Confusion Pairs (where model struggles) ---")
for i, true_cat in enumerate(label_order):
    for j, pred_cat in enumerate(label_order):
        if i != j and cm[i, j] > 5:
            print(f"  True: {true_cat:12s} → Predicted: {pred_cat:12s}  ({cm[i, j]} times)")

wrong_mask  = best_preds != y_test
wrong_texts = test_df["text"].values[wrong_mask]
wrong_true  = y_test[wrong_mask]
wrong_pred  = best_preds[wrong_mask]

print(f"\nMisclassified: {wrong_mask.sum()} / {len(y_test)}  "
      f"({100 * wrong_mask.sum() / len(y_test):.1f}%)")

print("\nSample misclassified articles:")
for i in range(min(3, len(wrong_texts))):
    snippet = wrong_texts[i][:180].replace("\n", " ")
    print(f"\n  True: {wrong_true[i]:12s} → Predicted: {wrong_pred[i]}")
    print(f"  \"{snippet}...\"")


# ─────────────────────────────────────────────
# Step 8: Model Comparison Chart
# ─────────────────────────────────────────────

summary = pd.DataFrame(
    {name: {"accuracy": res["accuracy"], "macro_f1": res["macro_f1"]}
     for name, res in results.items()}
).T.round(4)

print("\n--- Results Summary ---")
print(summary.to_string())

summary.to_csv("results/metrics.csv")

summary.plot(kind="bar", figsize=(10, 5), ylim=(0.80, 1.0), rot=20)
plt.title("Model Comparison — News Classifier")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig("results/model_comparison.png", dpi=150)
print("Saved → results/model_comparison.png")

print("\nDone. All outputs saved in results/")

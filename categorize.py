"""
categorize.py
--------------
Trains a text classifier on transaction Descriptions to predict Category,
using the rows that already have a Category as training data. Then uses
the trained model to fill in the 88 rows with missing Category (these were
left NaN by clean_data.py on purpose, since imputing a category from raw
text is a genuine ML task, not just data cleaning).

Approach: TF-IDF (character n-grams) + Linear SVM.
Why character n-grams instead of word-level: merchant names in this data
have casing/spacing noise ("SWIGGY", "swiggy  ", "Zomato*Order") that a
word-level vectorizer would treat as different tokens. Character n-grams
(e.g. "swig", "iggy") stay robust to that noise without needing manual
regex cleanup of every merchant name variant.

Run with:
    python categorize.py
Output: data/transactions_categorized.csv
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

CLEAN_PATH = "data/transactions_clean.csv"
OUTPUT_PATH = "data/transactions_categorized.csv"


def normalise_description(text: str) -> str:
    """Lowercase + strip common noise characters before vectorizing."""
    return str(text).lower().replace("*", " ").strip()


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)),
        ("clf", LinearSVC(random_state=42)),
    ])


def train_and_fill():
    df = pd.read_csv(CLEAN_PATH, parse_dates=["Date"])
    df["DescClean"] = df["Description"].apply(normalise_description)

    labeled = df[df["Category"].notna()].copy()
    unlabeled = df[df["Category"].isna()].copy()

    print(f"Labeled rows (training data): {len(labeled)}")
    print(f"Unlabeled rows (to predict): {len(unlabeled)}")

    X = labeled["DescClean"]
    y = labeled["Category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nHeld-out test accuracy: {acc:.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix plot — good evidence for the report
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Categorization Confusion Matrix (Test Accuracy: {acc:.1%})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("eda_plots/categorization_confusion_matrix.png", dpi=120)
    plt.close()
    print("Saved confusion matrix to eda_plots/categorization_confusion_matrix.png")

    # Retrain on ALL labeled data (train+test) before predicting the real
    # missing rows, so we're not throwing away 20% of our labeled signal.
    final_pipeline = build_pipeline()
    final_pipeline.fit(X, y)

    if len(unlabeled) > 0:
        predicted = final_pipeline.predict(unlabeled["DescClean"])
        df.loc[df["Category"].isna(), "Category"] = predicted
        df.loc[unlabeled.index, "CategoryPredicted"] = True
        print(f"\nFilled {len(unlabeled)} missing categories using the trained classifier")

    df["CategoryPredicted"] = df.get("CategoryPredicted", False).fillna(False)
    df = df.drop(columns=["DescClean"])
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved final categorized dataset to {OUTPUT_PATH}")

    return df, acc


if __name__ == "__main__":
    df, acc = train_and_fill()
    print("\nSample of previously-missing rows, now filled:")
    print(df[df["CategoryPredicted"] == True][["Description", "Category"]].head(10).to_string())

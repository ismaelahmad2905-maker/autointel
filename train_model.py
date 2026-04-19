import os
from pathlib import Path
import django
import joblib
import pandas as pd

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "autointel.settings")
django.setup()

from main.models import Problem
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():
    problems = Problem.objects.all()
    df = pd.DataFrame(list(problems.values("problem_text", "category")))

    if df.empty:
        print("No data found in Problem table.")
        return

    df["problem_text"] = df["problem_text"].astype(str).str.lower().str.strip()

    X = df["problem_text"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    joblib.dump(model, MODEL_DIR / "problem_classifier.joblib")
    print("Saved:", MODEL_DIR / "problem_classifier.joblib")


if __name__ == "__main__":
    main()
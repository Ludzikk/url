
from pathlib import Path
import json

import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==========================================
# KONFIGURACJA
# ==========================================

BASE = Path(__file__).resolve().parent

TRAIN_FILE = BASE / "split" / "train_80.csv"
TEST_FILE = BASE / "split" / "test_20.csv"

OUTPUT_DIR = BASE / "wyniki" / "tfidf_domeny"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


def main():

    # ==========================================
    # 1. WCZYTANIE DANYCH
    # ==========================================

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    print("Wczytano dane:")
    print(f"Trening: {len(train)} rekordów")
    print(f"Test: {len(test)} rekordów")

    # Sprawdzenie wymaganych kolumn
    for name, df in [("train", train), ("test", test)]:

        required = {"Domain", "label"}

        if not required.issubset(df.columns):
            raise ValueError(
                f"Brakuje kolumn w {name}: "
                f"{required - set(df.columns)}"
            )

        if df[["Domain", "label"]].isna().any().any():
            raise ValueError(
                f"Brakujące wartości Domain lub label w {name}"
            )

        if not df["label"].isin([0, 1]).all():
            raise ValueError(
                f"Nieprawidłowe etykiety w {name}"
            )

    # ==========================================
    # 2. SPRAWDZENIE POWTARZAJĄCYCH SIĘ DOMEN
    # ==========================================

    train_domains = set(
        train["Domain"].astype(str).str.strip().str.lower().str.rstrip(".")
    )

    test_domains = set(
        test["Domain"].astype(str).str.strip().str.lower().str.rstrip(".")
    )

    overlap = train_domains.intersection(test_domains)

    print(f"\nWspólne domeny: {len(overlap)}")

    if overlap:
        raise ValueError(
            "BŁĄD: Te same domeny występują w treningu i teście!"
        )

    print("OK - brak wspólnych domen.")

    # ==========================================
    # 3. PRZYGOTOWANIE DANYCH
    # ==========================================

    X_train = train["Domain"].astype(str)
    X_test = test["Domain"].astype(str)

    # PhiUSIIL:
    # 0 = phishing, 1 = legalny
    #
    # Zamiana:
    # 1 = phishing, 0 = legalny

    y_train = (train["label"] == 0).astype(int)
    y_test = (test["label"] == 0).astype(int)

    # ==========================================
    # 4. MODEL TF-IDF + LOGISTIC REGRESSION
    # ==========================================

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                lowercase=True,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=SEED
            )
        )
    ])

    # ==========================================
    # 5. TRENOWANIE MODELU
    # ==========================================

    print("\nTrenowanie modelu...")

    model.fit(X_train, y_train)

    print("Trenowanie zakończone.")

    # ==========================================
    # 6. PREDYKCJA
    # ==========================================

    y_pred = model.predict(X_test)

    # ==========================================
    # 7. OBLICZANIE METRYK
    # ==========================================

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    ).ravel()

    # ==========================================
    # 8. TABELA WYNIKÓW
    # ==========================================

    results = {
        "Accuracy": f"{accuracy * 100:.2f}%",
        "Precision": f"{precision * 100:.2f}%",
        "Recall": f"{recall * 100:.2f}%",
        "F1-score": f"{f1 * 100:.2f}%",
        "FP": int(fp),
        "FN": int(fn)
    }

    print("\n" + "=" * 36)
    print("TF-IDF + LOGISTIC REGRESSION")
    print("Dane wejściowe: Domain")
    print("=" * 36)

    print(f"{'Metryka':<18} | {'Wynik':>12}")
    print("-" * 33)

    for metric, value in results.items():
        print(f"{metric:<18} | {str(value):>12}")

    print("=" * 36)

    # ==========================================
    # 9. ZAPIS WYNIKÓW DO CSV
    # ==========================================

    results_df = pd.DataFrame(
        list(results.items()),
        columns=["Metryka", "Wynik"]
    )

    results_df.to_csv(
        OUTPUT_DIR / "tabela_wynikow.csv",
        index=False,
        sep=";",
        encoding="utf-8-sig"
    )

    # ==========================================
    # 10. ZAPIS WYNIKÓW DO JSON
    # ==========================================

    results_json = {
        "model": "TF-IDF + Logistic Regression",
        "features": "Domain",
        "train_size": len(train),
        "test_size": len(test),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "TN": int(tn)
    }

    with open(
        OUTPUT_DIR / "wyniki.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            results_json,
            file,
            indent=4,
            ensure_ascii=False
        )

    # ==========================================
    # 11. ZAPIS MODELU
    # ==========================================

    joblib.dump(
        model,
        OUTPUT_DIR / "model.joblib"
    )

    print(f"\nZapisano wyniki w: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


from pathlib import Path
from urllib.parse import urlsplit
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

OUTPUT_DIR = BASE / "wyniki" / "tfidf_sciezka_parametry"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


# ==========================================
# 1. EKSTRAKCJA SCIEZKI I PARAMETROW
# ==========================================

def extract_path_query(url):

    url = str(url).strip()

    # Obsluga adresow bez schematu
    if url.startswith("//"):
        url = "http:" + url
    elif "://" not in url:
        url = "http://" + url

    parsed = urlsplit(url)

    path = parsed.path or "/"
    query = parsed.query

    if query:
        return f"{path}?{query}"

    return path


def main():

    # ==========================================
    # 2. WCZYTANIE DANYCH
    # ==========================================

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    print("Wczytano dane:")
    print(f"Trening: {len(train)} rekordow")
    print(f"Test: {len(test)} rekordow")

    for name, df in [("train", train), ("test", test)]:

        required = {"URL", "Domain", "label"}

        if not required.issubset(df.columns):
            raise ValueError(
                f"Brakuje kolumn w {name}: "
                f"{required - set(df.columns)}"
            )

        if df[list(required)].isna().any().any():
            raise ValueError(
                f"Brakujace dane w {name}"
            )

        if not df["label"].isin([0, 1]).all():
            raise ValueError(
                f"Nieprawidlowe etykiety w {name}"
            )

    # ==========================================
    # 3. SPRAWDZENIE WSPOLNYCH DOMEN
    # ==========================================

    train_domains = set(
        train["Domain"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.rstrip(".")
    )

    test_domains = set(
        test["Domain"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.rstrip(".")
    )

    overlap = train_domains & test_domains

    print(f"\nWspolne domeny: {len(overlap)}")

    if overlap:
        raise ValueError(
            "BLAD: Domeny powtarzaja sie w obu zbiorach!"
        )

    print("OK - brak wspolnych domen.")

    # ==========================================
    # 4. PRZYGOTOWANIE DANYCH
    # ==========================================

    X_train = train["URL"].apply(extract_path_query)
    X_test = test["URL"].apply(extract_path_query)

    # PhiUSIIL:
    # 0 = phishing
    # 1 = legalny
    #
    # Zamiana:
    # 1 = phishing
    # 0 = legalny

    y_train = (train["label"] == 0).astype(int)
    y_test = (test["label"] == 0).astype(int)

    print("\nPrzykladowe dane po ekstrakcji:")
    print(X_train.head(5).to_string(index=False))

    # ==========================================
    # 5. MODEL TF-IDF + LOGISTIC REGRESSION
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
    # 6. TRENOWANIE MODELU
    # ==========================================

    print("\nTrenowanie modelu...")

    model.fit(X_train, y_train)

    print("Trenowanie zakonczone.")

    # ==========================================
    # 7. PREDYKCJA
    # ==========================================

    y_pred = model.predict(X_test)

    # ==========================================
    # 8. OBLICZANIE METRYK
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
    # 9. TABELA WYNIKOW
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
    print("Dane: Sciezka + parametry URL")
    print("=" * 36)

    print(f"{'Metryka':<18} | {'Wynik':>12}")
    print("-" * 33)

    for metric, value in results.items():
        print(f"{metric:<18} | {str(value):>12}")

    print("=" * 36)

    # ==========================================
    # 10. ZAPIS TABELI CSV
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
    # 11. ZAPIS JSON
    # ==========================================

    results_json = {
        "model": "TF-IDF + Logistic Regression",
        "features": "URL path + query",
        "random_seed": SEED,
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
    # 12. ZAPIS MODELU
    # ==========================================

    joblib.dump(
        model,
        OUTPUT_DIR / "model.joblib"
    )

    print(f"\nZapisano wyniki w: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

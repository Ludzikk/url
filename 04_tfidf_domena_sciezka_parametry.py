
from pathlib import Path
from urllib.parse import urlsplit
import json

import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
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

OUTPUT_DIR = BASE / "wyniki" / "tfidf_domena_sciezka_parametry"

SEED = 42


# ==========================================
# 1. EKSTRAKCJA SCIEZKI I PARAMETROW
# ==========================================

def extract_path_query(url):
    url = str(url).strip()

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


# ==========================================
# 2. WALIDACJA DANYCH
# ==========================================

def check_data(df, name):

    required = {"URL", "Domain", "label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Brakuje kolumn w {name}: {sorted(missing)}"
        )

    if df[["URL", "Domain", "label"]].isna().any().any():
        raise ValueError(
            f"Brakujace wartosci w {name}"
        )

    if not df["label"].isin([0, 1]).all():
        raise ValueError(
            f"Nieprawidlowe etykiety w {name}"
        )


def normalized_domains(df):
    return set(
        df["Domain"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.rstrip(".")
    )


def main():

    # ==========================================
    # 3. WCZYTANIE DANYCH
    # ==========================================

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    check_data(train, "train_80.csv")
    check_data(test, "test_20.csv")

    print(f"Trening: {len(train)} rekordow")
    print(f"Test: {len(test)} rekordow")

    # ==========================================
    # 4. SPRAWDZENIE WSPOLNYCH DOMEN
    # ==========================================

    overlap = (
        normalized_domains(train)
        & normalized_domains(test)
    )

    print(f"Wspolne domeny: {len(overlap)}")

    if overlap:
        raise ValueError(
            "BLAD: Te same domeny wystepuja "
            "w treningu i tescie!"
        )

    print("OK - brak wspolnych domen.")

    # ==========================================
    # 5. PRZYGOTOWANIE DANYCH
    # ==========================================

    X_train = pd.DataFrame({
        "Domain": train["Domain"].astype(str),
        "path_query": train["URL"].map(extract_path_query)
    })

    X_test = pd.DataFrame({
        "Domain": test["Domain"].astype(str),
        "path_query": test["URL"].map(extract_path_query)
    })

    # PhiUSIIL:
    # 0 = phishing, 1 = legalny
    #
    # Po zamianie:
    # 1 = phishing, 0 = legalny

    y_train = (train["label"] == 0).astype(int)
    y_test = (test["label"] == 0).astype(int)

    # ==========================================
    # 6. DWA ODDZIELNE WEKTORYZATORY TF-IDF
    # ==========================================

    features = ColumnTransformer(
        transformers=[
            (
                "domain_tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    lowercase=True,
                    sublinear_tf=True
                ),
                "Domain"
            ),
            (
                "path_query_tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    lowercase=True,
                    sublinear_tf=True
                ),
                "path_query"
            )
        ],
        sparse_threshold=1.0
    )

    # ==========================================
    # 7. MODEL LOGISTIC REGRESSION
    # ==========================================

    model = Pipeline([
        ("tfidf", features),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=SEED
            )
        )
    ])

    # ==========================================
    # 8. TRENOWANIE MODELU
    # ==========================================

    print("\nTrenowanie modelu...")

    model.fit(X_train, y_train)

    print("Trenowanie zakonczone.")

    # ==========================================
    # 9. PREDYKCJA
    # ==========================================

    y_pred = model.predict(X_test)

    # ==========================================
    # 10. OBLICZANIE METRYK
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
    # 11. TABELA WYNIKOW
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
    print("Dane: Domain + sciezka i parametry")
    print("=" * 36)

    print(f"{'Metryka':<18} | {'Wynik':>12}")
    print("-" * 33)

    for metric, value in results.items():
        print(f"{metric:<18} | {str(value):>12}")

    print("=" * 36)

    # ==========================================
    # 12. ZAPIS WYNIKOW CSV
    # ==========================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

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
    # 13. ZAPIS WYNIKOW JSON
    # ==========================================

    results_json = {
        "model": "TF-IDF + Logistic Regression",
        "features": "Domain + path/query",
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
    # 14. ZAPIS MODELU
    # ==========================================

    joblib.dump(
        model,
        OUTPUT_DIR / "model.joblib"
    )

    print(f"\nZapisano wyniki w: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

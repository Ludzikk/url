"""Eksperyment 4: cechy leksykalne osobno i w polaczeniu z TF-IDF domeny oraz sciezki+parametrow.

Wejscie: split/train_80.csv i split/test_20.csv (podzial nie jest modyfikowany).
Wyniki: wyniki/cechy_i_tfidf/porownanie.csv + tabele/<nazwa>.csv.
Klasa 1 to phishing (oryginalna etykieta PhiUSIIL 0).
"""

from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
TRAIN_FILE = BASE / "split" / "train_80.csv"
TEST_FILE = BASE / "split" / "test_20.csv"
OUTPUT = BASE / "wyniki" / "cechy_i_tfidf"
SEED = 42

# Ta sama lista 19 cech co we wczesniejszym eksperymencie leksykalnym.
CECHY = [
    "URLLength", "DomainLength", "IsDomainIP", "CharContinuationRate",
    "TLDLength", "NoOfSubDomain", "HasObfuscation", "NoOfObfuscatedChar",
    "ObfuscationRatio", "NoOfLettersInURL", "LetterRatioInURL",
    "NoOfDegitsInURL", "DegitRatioInURL", "NoOfEqualsInURL",
    "NoOfQMarkInURL", "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "IsHTTPS",
]


def sciezka_parametry(adres):
    adres = str(adres).strip()
    if adres.startswith("//"):
        adres = "http:" + adres
    elif "://" not in adres:
        adres = "http://" + adres
    parsed = urlsplit(adres)
    sciezka = parsed.path or "/"
    return sciezka + ("?" + parsed.query if parsed.query else "")


def normalizuj_domeny(df):
    return set(df["Domain"].astype(str).str.strip().str.lower().str.rstrip("."))


def policz_metryki(y_true, y_pred, nazwa):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Eksperyment": nazwa,
        "Accuracy": round(100 * accuracy_score(y_true, y_pred), 2),
        "Precision": round(100 * precision_score(y_true, y_pred, zero_division=0), 2),
        "Recall": round(100 * recall_score(y_true, y_pred, zero_division=0), 2),
        "F1-score": round(100 * f1_score(y_true, y_pred, zero_division=0), 2),
        "FP": int(fp),
        "FN": int(fn),
    }


def pokaz_tabele(w):
    print("\n" + "=" * 54)
    print(w["Eksperyment"])
    print("-" * 34)
    print(f"{'Metryka':<18} | {'Wynik':>12}")
    print("-" * 34)
    for metryka in ["Accuracy", "Precision", "Recall", "F1-score", "FP", "FN"]:
        wartosc = f"{w[metryka]:.2f}%" if metryka not in ("FP", "FN") else str(w[metryka])
        print(f"{metryka:<18} | {wartosc:>12}")


def naucz_ocen(X_train, X_test, y_train, y_test, nazwa):
    klasyfikator = LogisticRegression(max_iter=2000, solver="liblinear", random_state=SEED)
    klasyfikator.fit(X_train, y_train)
    return policz_metryki(y_test, klasyfikator.predict(X_test), nazwa)


def main():
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    wymagane = set(CECHY) | {"Domain", "URL", "label"}
    for nazwa, df in [("train", train), ("test", test)]:
        brak = wymagane - set(df.columns)
        if brak:
            raise ValueError(f"Brak kolumn w {nazwa}: {sorted(brak)}")
        if df[list(wymagane)].isna().any().any():
            raise ValueError(f"Brakujace wartosci w wymaganych kolumnach: {nazwa}")
        if not df["label"].isin([0, 1]).all():
            raise ValueError(f"Nieprawidlowe etykiety w {nazwa}")

    wspolne = normalizuj_domeny(train) & normalizuj_domeny(test)
    print(f"Trening: {len(train)}, test: {len(test)}, wspolne domeny: {len(wspolne)}")
    if wspolne:
        raise ValueError("Powtarzajace sie domeny miedzy treningiem a testem!")

    # 0 w PhiUSIIL = phishing. W metrykach 1 = phishing.
    y_train = (train["label"] == 0).astype(int)
    y_test = (test["label"] == 0).astype(int)

    # Standaryzacja wylacznie na treningu (dla wszystkich cech numerycznych).
    liczby_train = train[CECHY].apply(pd.to_numeric, errors="raise")
    liczby_test = test[CECHY].apply(pd.to_numeric, errors="raise")
    scaler = StandardScaler()
    X_liczby_train = scaler.fit_transform(liczby_train)
    X_liczby_test = scaler.transform(liczby_test)

    # TF-IDF dla domeny i sciezki+parametrow: osobne slowniki, uczone tylko na treningu.
    domena_vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), lowercase=True, sublinear_tf=True
    )
    sciezka_vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), lowercase=True, sublinear_tf=True
    )
    X_domena_train = domena_vectorizer.fit_transform(train["Domain"].astype(str))
    X_domena_test = domena_vectorizer.transform(test["Domain"].astype(str))
    X_sciezka_train = sciezka_vectorizer.fit_transform(train["URL"].map(sciezka_parametry))
    X_sciezka_test = sciezka_vectorizer.transform(test["URL"].map(sciezka_parametry))
    X_tekst_train = hstack([X_domena_train, X_sciezka_train], format="csr")
    X_tekst_test = hstack([X_domena_test, X_sciezka_test], format="csr")

    wyniki = []

    def uruchom(nazwa, a, b):
        print(f"Uczenie: {nazwa} ...", flush=True)
        wynik = naucz_ocen(a, b, y_train, y_test, nazwa)
        wyniki.append(wynik)
        pokaz_tabele(wynik)

    # Punkt odniesienia: tylko tekst. Dla uczciwego porownania we wszystkich wariantach
    # stosujemy solver="liblinear" (poprzedni E3 uzywal domyslnego solvera).
    uruchom("TFIDF_domena_sciezka_parametry", X_tekst_train, X_tekst_test)

    # A: kazda z 19 cech numerycznych osobno (bez TF-IDF).
    for i, cecha in enumerate(CECHY):
        uruchom(cecha, X_liczby_train[:, [i]], X_liczby_test[:, [i]])

    # B: oddzielna cecha numeryczna + ten sam TF-IDF domeny oraz sciezki/parametrow.
    for i, cecha in enumerate(CECHY):
        a = hstack([X_tekst_train, csr_matrix(X_liczby_train[:, [i]])], format="csr")
        b = hstack([X_tekst_test, csr_matrix(X_liczby_test[:, [i]])], format="csr")
        uruchom(f"TFIDF_plus_{cecha}", a, b)

    # C: caly pakiet 19 cech numerycznych bez TF-IDF oraz z TF-IDF.
    uruchom("Wszystkie_19_cech", X_liczby_train, X_liczby_test)
    uruchom(
        "TFIDF_plus_wszystkie_19_cech",
        hstack([X_tekst_train, csr_matrix(X_liczby_train)], format="csr"),
        hstack([X_tekst_test, csr_matrix(X_liczby_test)], format="csr"),
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(wyniki).to_csv(
        OUTPUT / "porownanie.csv", sep=";", encoding="utf-8-sig", index=False
    )
    # Osobna tabela Metryka | Wynik dla kazdego wariantu.
    folder_tabel = OUTPUT / "tabele"
    folder_tabel.mkdir(exist_ok=True)
    for w in wyniki:
        dane = [
            (k, f"{w[k]:.2f}%" if k not in ("FP", "FN") else w[k])
            for k in ["Accuracy", "Precision", "Recall", "F1-score", "FP", "FN"]
        ]
        pd.DataFrame(dane, columns=["Metryka", "Wynik"]).to_csv(
            folder_tabel / f"{w['Eksperyment']}.csv",
            sep=";", encoding="utf-8-sig", index=False,
        )

    print(f"\nWykonano {len(wyniki)} eksperymentow.")
    print(f"Zapisano zestawienie: {OUTPUT / 'porownanie.csv'}")
    print(f"Zapisano pojedyncze tabele: {folder_tabel}")


if __name__ == "__main__":
    main()

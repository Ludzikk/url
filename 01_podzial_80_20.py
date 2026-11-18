"""Podział PhiUSIIL na train/test bez wspólnych domen (seed=42)."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

BASE = Path(__file__).resolve().parent
INPUT = BASE / 'data' / 'phiusiil_10000.csv'
OUTPUT = BASE / 'split'
SEED = 42

def main():
    df = pd.read_csv(INPUT)
    required = {'Domain', 'label'}
    if not required.issubset(df.columns):
        raise ValueError(f'Brakuje kolumn: {sorted(required - set(df.columns))}')
    if df[['Domain', 'label']].isna().any().any():
        raise ValueError('W kolumnie Domain lub label są braki danych.')

    # Normalizacja służy TYLKO do wyznaczenia grup, nie modyfikuje oryginalnych danych.
    groups = df['Domain'].astype(str).str.strip().str.lower().str.rstrip('.')
    y = df['label']
    overall_ratio = y.value_counts(normalize=True)
    splitter = GroupShuffleSplit(n_splits=150, test_size=0.20, random_state=SEED)

    def score(pair):
        train_idx, test_idx = pair
        test_ratio = y.iloc[test_idx].value_counts(normalize=True)
        class_difference = sum(abs(test_ratio.get(c, 0) - p) for c, p in overall_ratio.items())
        size_difference = abs(len(test_idx) / len(df) - 0.20)
        return size_difference + class_difference

    # Wybieramy powtarzalny podział możliwie bliski 80/20 i proporcjom klas.
    train_idx, test_idx = min(splitter.split(df, y, groups), key=score)
    train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    overlap = train_groups & test_groups
    if overlap:
        raise AssertionError(f'BŁĄD: {len(overlap)} domen występuje w obu zbiorach!')

    OUTPUT.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUTPUT / 'train_80.csv', index=False, encoding='utf-8-sig')
    test.to_csv(OUTPUT / 'test_20.csv', index=False, encoding='utf-8-sig')
    print(f'Trening: {len(train)} rekordów | rozkład klas: {train.label.value_counts().to_dict()}')
    print(f'Test: {len(test)} rekordów | rozkład klas: {test.label.value_counts().to_dict()}')
    print(f'Wspólne domeny: {len(overlap)}')
    print(f'Zapisano: {OUTPUT / "train_80.csv"} oraz {OUTPUT / "test_20.csv"}')

if __name__ == '__main__':
    main()

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from models import build_classifiers, extract_features
from preprocessing import DATASET_PATH

WINDOW = 2
MIN_SENSE_COUNT = 5


def evaluate(df, window):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for word, group in df.groupby('word'):
        X = [
            extract_features(tokens.split(), pos.split(), position, window)
            for tokens, pos, position in zip(group.tokens, group.pos, group.position)
        ]
        y = group.sense
        for name, clf in build_classifiers().items():
            pred = cross_val_predict(clf, X, y, cv=folds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y, pred, average='macro', zero_division=0
            )
            rows.append({
                'word': word,
                'model': name,
                'accuracy': accuracy_score(y, pred),
                'precision': precision,
                'recall': recall,
                'f1': f1,
            })
    return pd.DataFrame(rows)


def drop_rare_senses(df, min_count):
    counts = df.groupby(['word', 'sense']).sense.transform('size')
    return df[counts >= min_count]


if __name__ == '__main__':
    df = drop_rare_senses(pd.read_csv(DATASET_PATH), MIN_SENSE_COUNT)
    print(evaluate(df, WINDOW).round(3).to_string(index=False))

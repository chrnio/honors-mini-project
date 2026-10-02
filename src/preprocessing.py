from pathlib import Path

import pandas as pd
from nltk.corpus import semcor
from nltk.corpus.reader.wordnet import Lemma
from nltk.tree import Tree

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / 'data' / 'annotated' / 'semcor_wsd.csv'

TARGETS = {'bank', 'interest', 'plant', 'run'}


def load_semcor_instances(targets):
    rows = []
    for sent in semcor.tagged_sents(tag='sem'):
        tokens, hits = [], []
        for chunk in sent:
            if isinstance(chunk, Tree):
                label = chunk.label()
                if isinstance(label, Lemma) and label.name().lower() in targets:
                    hits.append((len(tokens), label))
                tokens.extend(chunk.leaves())
            else:
                tokens.extend(chunk)

        for position, lemma in hits:
            rows.append({
                'word': lemma.name().lower(),
                'sense': lemma.synset().name(),
                'position': position,
                'tokens': ' '.join(tokens),
            })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = load_semcor_instances(TARGETS)
    df.to_csv(DATASET_PATH, index=False)
    print(df.groupby('word').size())

from pathlib import Path

import pandas as pd
from nltk import pos_tag
from nltk.corpus import semcor, stopwords
from nltk.corpus.reader.wordnet import Lemma
from nltk.stem import WordNetLemmatizer
from nltk.tree import Tree

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / 'data' / 'annotated' / 'semcor_wsd.csv'

TARGETS = {'bank', 'interest', 'plant', 'run'}

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def wordnet_pos(tag):
    return {'J': 'a', 'V': 'v', 'N': 'n', 'R': 'r'}.get(tag[0], 'n')


def normalize(token, tag):
    """Lowercased lemma of a token, or None for stopwords and punctuation."""
    word = token.lower()
    if not word.isalpha() or word in STOPWORDS:
        return None
    return lemmatizer.lemmatize(word, wordnet_pos(tag))


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

        if not hits:
            continue
        tags = [tag for _, tag in pos_tag(tokens)]
        for position, lemma in hits:
            rows.append({
                'word': lemma.name().lower(),
                'sense': lemma.synset().name(),
                'position': position,
                'tokens': ' '.join(tokens),
                'pos': ' '.join(tags),
            })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = load_semcor_instances(TARGETS)
    df.to_csv(DATASET_PATH, index=False)
    print(df.groupby('word').size())

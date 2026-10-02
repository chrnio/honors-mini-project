from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from preprocessing import normalize


def extract_features(tokens, tags, position, window):
    """Target POS, neighbouring POS tags and a bag-of-words within +-window tokens."""
    features = {'pos_0': tags[position]}
    for offset in range(-window, window + 1):
        i = position + offset
        if offset == 0 or not 0 <= i < len(tokens):
            continue
        features[f'pos_{offset:+d}'] = tags[i]
        lemma = normalize(tokens[i], tags[i])
        if lemma:
            features[f'bow_{lemma}'] = features.get(f'bow_{lemma}', 0) + 1
    return features


def build_classifiers():
    return {
        'SVM': make_pipeline(DictVectorizer(), SVC(kernel='linear')),
        'Naive Bayes': make_pipeline(DictVectorizer(), MultinomialNB()),
    }

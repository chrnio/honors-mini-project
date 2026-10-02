# Supervised Machine Learning Approach for WSD in Spoken News

## Application Context 
News broadcast transcript processing requires high accuracy when disambiguating frequent polysemous terms (e.g., "bank", "interest", "plant", "run").

## Core Concepts
- Supervised WSD 
- Local Context Features
- POS Tag Features
- Surrounding Window Bag-of-Words
- Classifier Training

## Task Description
Build a supervised WSD dataset (using Senseval/SemCor or custom labeled transcript sentences). Extract feature vectors consisting of target word POS tags, neighboring POS tags, and a context window Bag-of-Words. Train Support Vector Machine (SVM) and Naive Bayes classifiers to predict synset labels.

## Suggested Libraries
- `nltk.corpus.senseval`
- `scikit-learn (SVC, RandomForestClassifier)`
- `pandas`

## Deliverables:
1. ML documentation covering feature vector construction, window size impact (W=±2 vs W= ± 5 ), and classification metric analysis.
2. Python source code for training and testing the supervised WSD models.

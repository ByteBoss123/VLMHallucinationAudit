import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def load(path):
    return [json.loads(l) for l in open(path)]

train = load('data/train_sentences.jsonl')
X_train, y_train = [r['text'] for r in train], [r['label'] for r in train]

vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
Xtr = vec.fit_transform(X_train)
clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
clf.fit(Xtr, y_train)

feature_names = vec.get_feature_names_out()
incorrect_idx = list(clf.classes_).index('incorrect')
coefs = clf.coef_[0] if len(clf.classes_) == 2 else clf.coef_[incorrect_idx]
# For binary LogisticRegression, coef_ is shape (1, n_features), positive = toward classes_[1]
# classes_ is alphabetical: ['correct', 'incorrect'], so positive coef -> 'incorrect'
top_hallucination_signals = sorted(zip(feature_names, coefs), key=lambda x: -x[1])[:20]
top_correct_signals = sorted(zip(feature_names, coefs), key=lambda x: x[1])[:20]

print('=== Top n-grams predicting HALLUCINATION ===')
for term, w in top_hallucination_signals:
    print(f'  {term:<30} {w:.3f}')

print()
print('=== Top n-grams predicting CORRECT (grounded) ===')
for term, w in top_correct_signals:
    print(f'  {term:<30} {w:.3f}')

output = {
    'top_hallucination_ngrams': [{'term': t, 'weight': round(float(w), 4)} for t, w in top_hallucination_signals],
    'top_correct_ngrams': [{'term': t, 'weight': round(float(w), 4)} for t, w in top_correct_signals],
}
with open('results/interpretability.json', 'w') as f:
    json.dump(output, f, indent=2)

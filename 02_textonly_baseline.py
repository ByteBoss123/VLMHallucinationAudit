import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.dummy import DummyClassifier

def load(path):
    return [json.loads(l) for l in open(path)]

train = load('data/train_sentences.jsonl')
test = load('data/test_sentences.jsonl')

X_train, y_train = [r['text'] for r in train], [r['label'] for r in train]
X_test, y_test = [r['text'] for r in test], [r['label'] for r in test]

# Baseline: majority class
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
dummy_f1 = f1_score(y_test, dummy_pred, pos_label='incorrect')

# Text-only TF-IDF + Logistic Regression
vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
Xtr = vec.fit_transform(X_train)
Xte = vec.transform(X_test)

clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
clf.fit(Xtr, y_train)
pred = clf.predict(Xte)
proba = clf.predict_proba(Xte)[:, list(clf.classes_).index('incorrect')]

auc = roc_auc_score([1 if l == 'incorrect' else 0 for l in y_test], proba)
f1 = f1_score(y_test, pred, pos_label='incorrect')

print('=== Majority-class baseline ===')
print('F1 (incorrect class):', round(dummy_f1, 4))
print()
print('=== TF-IDF + Logistic Regression (text-only) ===')
print('ROC-AUC:', round(auc, 4))
print('F1 (incorrect class):', round(f1, 4))
print()
print(classification_report(y_test, pred))

results = {
    'hypothesis': 'Text-only features cannot fully detect visual hallucination without image grounding; '
                  'this tests how far surface language alone gets.',
    'n_train_sentences': len(train), 'n_test_sentences': len(test),
    'test_class_balance': {'correct': y_test.count('correct'), 'incorrect': y_test.count('incorrect')},
    'majority_baseline_f1_incorrect': round(dummy_f1, 4),
    'textonly_roc_auc': round(auc, 4),
    'textonly_f1_incorrect': round(f1, 4),
}
with open('results/textonly_classifier_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

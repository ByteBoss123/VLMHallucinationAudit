"""
VLMHallucinationAudit — serving layer.

Scores a single VLM-generated sentence for hallucination risk using the
text-only TF-IDF + Logistic Regression model. Intended as a fast pre-filter
layer in front of a full multimodal (image-grounded) verifier, not a
replacement for one — see README for the disclosed scope limitation.
"""
import json
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = FastAPI(title="VLMHallucinationAudit")


def load(path):
    return [json.loads(l) for l in open(path)]


_train = load('data/train_sentences.jsonl')
_vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
_Xtr = _vec.fit_transform([r['text'] for r in _train])
_clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
_clf.fit(_Xtr, [r['label'] for r in _train])

_incorrect_idx = list(_clf.classes_).index('incorrect')


class ScoreRequest(BaseModel):
    sentence: str


class ScoreResponse(BaseModel):
    hallucination_risk: float
    prediction: str
    flag_for_review: bool


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    X = _vec.transform([req.sentence])
    proba = _clf.predict_proba(X)[0]
    risk = float(proba[_incorrect_idx])
    pred = _clf.predict(X)[0]
    return ScoreResponse(
        hallucination_risk=round(risk, 4),
        prediction=pred,
        flag_for_review=risk > 0.5,
    )

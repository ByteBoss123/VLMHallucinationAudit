import json
from collections import Counter

all_ann = json.load(open('data/all_annotations.json'))
train_ids = set(json.load(open('data/train.json')).keys())
test_ids = set(json.load(open('data/test.json')).keys())

def extract_sentences(ids):
    rows = []
    for img_id in ids:
        if img_id not in all_ann:
            continue
        entry = all_ann[img_id]
        for span_key, ann in entry['annotations'].items():
            label = ann['type']  # 'correct' or 'incorrect'
            if label not in ('correct', 'incorrect'):
                continue
            rows.append({'image_id': img_id, 'text': ann['text'], 'label': label})
    return rows

train_rows = extract_sentences(train_ids)
test_rows = extract_sentences(test_ids)

with open('data/train_sentences.jsonl', 'w') as f:
    for r in train_rows:
        f.write(json.dumps(r) + '\n')
with open('data/test_sentences.jsonl', 'w') as f:
    for r in test_rows:
        f.write(json.dumps(r) + '\n')

print('train sentences:', len(train_rows), dict(Counter(r['label'] for r in train_rows)))
print('test sentences:', len(test_rows), dict(Counter(r['label'] for r in test_rows)))

# recommender-cf

Item-based collaborative filtering (cosine similarity on a user-item interaction matrix) compared with a most-popular baseline, evaluated with leave-one-out hit rate@5.

## Run
```bash
pip install -r requirements.txt
python -m pytest -q   # 4 tests
python run.py
```

## Measured result (23 Nov 2026)
200 users, 60 items, 199 users evaluated (one interaction hidden each):

| method | hit rate@5 |
|---|---|
| item-based CF | 0.377 |
| most popular | 0.065 |

## Limits
Synthetic data where users belong to 4 clear taste groups, so CF has an easy job. Binary interactions only, no ratings, no cold-start handling, no time-aware split, dense matrix does not scale. One seed.

## Provenance
A learning and portfolio project.

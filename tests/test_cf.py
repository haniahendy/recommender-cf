import numpy as np

from rec.cf import item_similarity, leave_one_out_hit_rate, make_ratings, recommend


def test_similarity_symmetric_zero_diag():
    R, _, _ = make_ratings()
    S = item_similarity(R)
    assert np.allclose(S, S.T) and np.allclose(np.diag(S), 0)


def test_recommend_excludes_seen_items():
    R, _, _ = make_ratings()
    S = item_similarity(R)
    recs = recommend(R[0], S, 10)
    assert all(R[0][i] == 0 for i in recs) and len(set(recs)) == 10


def test_recommend_tiny_example():
    R = np.array([[1, 1, 0], [1, 1, 1], [0, 0, 1.0]])
    S = item_similarity(R)
    assert recommend(np.array([1, 0, 0.0]), S, 1) == [1]


def test_cf_beats_popularity_on_taste_group_data():
    R, _, _ = make_ratings()
    r = leave_one_out_hit_rate(R, 5)
    assert r["cf_hit_rate"] > r["popularity_hit_rate"]

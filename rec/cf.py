import numpy as np


def make_ratings(n_users: int = 200, n_items: int = 60, n_tastes: int = 4, seed: int = 8):
    """Binary interactions. Users belong to a taste group and mostly interact with that group's items."""
    rng = np.random.default_rng(seed)
    user_taste = rng.integers(0, n_tastes, n_users)
    item_taste = np.arange(n_items) % n_tastes
    p = np.where(user_taste[:, None] == item_taste[None, :], 0.35, 0.03)
    return (rng.random((n_users, n_items)) < p).astype(float), user_taste, item_taste


def item_similarity(R: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(R, axis=0, keepdims=True)
    norms[norms == 0] = 1
    S = (R / norms).T @ (R / norms)
    np.fill_diagonal(S, 0)
    return S


def recommend(R_user: np.ndarray, S: np.ndarray, k: int = 5) -> list[int]:
    scores = S @ R_user
    scores[R_user > 0] = -np.inf
    return [int(i) for i in np.argsort(scores)[::-1][:k]]


def popularity_recommend(R: np.ndarray, R_user: np.ndarray, k: int = 5) -> list[int]:
    pop = R.sum(0).astype(float)
    pop[R_user > 0] = -np.inf
    return [int(i) for i in np.argsort(pop)[::-1][:k]]


def leave_one_out_hit_rate(R: np.ndarray, k: int = 5, seed: int = 0) -> dict:
    """Hide one interacted item per user (users with >= 2 interactions), see if it is in the top k."""
    rng = np.random.default_rng(seed)
    held = {}
    train = R.copy()
    for u in range(R.shape[0]):
        items = np.flatnonzero(R[u])
        if len(items) >= 2:
            h = int(rng.choice(items))
            held[u] = h
            train[u, h] = 0
    S = item_similarity(train)
    cf = sum(h in recommend(train[u], S, k) for u, h in held.items())
    pop = sum(h in popularity_recommend(train, train[u], k) for u, h in held.items())
    n = len(held)
    return {"users_evaluated": n, "cf_hit_rate": round(cf / n, 3), "popularity_hit_rate": round(pop / n, 3)}

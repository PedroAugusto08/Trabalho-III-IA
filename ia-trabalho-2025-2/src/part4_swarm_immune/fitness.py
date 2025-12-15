from typing import List
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


#debug
print("fitness.py carregado")

Mask = List[int]

def make_fitness(X, y, alpha: float = 0.9, cv_splits: int = 3, seed: int = 42):
    X = np.asarray(X)
    y = np.asarray(y)
    d = X.shape[1]

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=seed
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000))
    ])


    def fitness(mask: Mask) -> float:
        idx = [i for i, b in enumerate(mask) if b == 1]
        if len(idx) == 0:
            return 1e9

        Xs = X[:, idx]
        acc = cross_val_score(clf, Xs, y, cv=cv, n_jobs=-1).mean()

        return alpha * (1.0 - acc) + (1.0 - alpha) * (len(idx) / d)

    return fitness

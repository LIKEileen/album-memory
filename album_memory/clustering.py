from __future__ import annotations

import numpy as np


def adaptive_similarity_threshold(
    similarities: list[float],
    *,
    fallback: float,
    floor: float,
    min_gap: float = 0.05,
) -> float:
    if len(similarities) < 4:
        return max(fallback, floor)
    try:
        from sklearn.mixture import GaussianMixture

        values = np.asarray(similarities, dtype=float).reshape(-1, 1)
        model = GaussianMixture(n_components=2, random_state=0, n_init=2)
        model.fit(values)
        means = np.sort(model.means_.ravel())
        if means[-1] - means[0] < min_gap:
            return max(fallback, floor)
        return max(float((means[-1] + means[0]) / 2), floor)
    except Exception:
        return max(fallback, floor)

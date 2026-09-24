from __future__ import annotations

import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def run_tsne(features: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    scaled = StandardScaler().fit_transform(features)
    effective_perplexity = min(perplexity, max(2.0, (len(scaled) - 1) / 3))
    return TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="random",
        learning_rate="auto",
        max_iter=1000,
        random_state=seed,
    ).fit_transform(scaled)

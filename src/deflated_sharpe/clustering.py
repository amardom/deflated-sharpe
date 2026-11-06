import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from threadpoolctl import threadpool_limits

def deduplicate_trials(values):
    first_indices = []
    labels = []
    tolerance = 4 * np.finfo(float).eps
    for column in range(values.shape[1]):
        match = next((group for group, first in enumerate(first_indices)
                      if np.all(np.abs(values[:, column] - values[:, first]) <= tolerance)), None)
        if match is None:
            match = len(first_indices)
            first_indices.append(column)
        labels.append(match)
    return values[:, first_indices], np.asarray(labels, dtype=int)

def cluster_trials(values):
    count = values.shape[1]
    correlation = np.clip(np.corrcoef(values, rowvar=False), -1, 1)
    correlation[np.isclose(correlation, 1, rtol=0, atol=1e-12)] = 1
    distance = np.sqrt((1 - correlation) / 2)
    np.fill_diagonal(distance, 0)
    if np.max(distance) < 1e-7:
        return np.zeros(count, dtype=int), None
    if count == 2:
        return np.arange(count), None
    best_quality = -np.inf
    best_labels = None
    with threadpool_limits(limits=1):
        unique_count = len(np.unique(np.round(distance, 12), axis=0))
        for groups in range(2, min(count - 1, unique_count) + 1):
            for seed in range(5):
                labels = KMeans(n_clusters=groups, n_init=1, random_state=seed).fit_predict(distance)
                if len(np.unique(labels)) != groups:
                    continue
                scores = silhouette_samples(distance, labels)
                quality = float(scores.mean() / max(scores.std(), 1e-12))
                if quality > best_quality:
                    best_quality = quality
                    best_labels = labels
    return best_labels, best_quality

def cluster_sharpes(values, labels):
    representatives = []
    for group in np.unique(labels):
        members, _ = deduplicate_trials(values[:, labels == group])
        members = members / members.std(axis=0, ddof=1)
        covariance = np.atleast_2d(np.cov(members, rowvar=False, ddof=1))
        inverse = np.linalg.pinv(covariance, rcond=1e-12)
        weights = inverse @ np.ones(members.shape[1])
        total = weights.sum()
        if abs(total) < 1e-12:
            raise ValueError("cluster has no stable fully invested minimum-variance representative")
        representative = members @ (weights / total)
        volatility = representative.std(ddof=1)
        if volatility < 1e-12:
            raise ValueError("cluster representative has zero variance; inspect opposing trials")
        representatives.append(float(representative.mean() / volatility))
    return representatives

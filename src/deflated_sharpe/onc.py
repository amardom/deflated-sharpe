import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from threadpoolctl import threadpool_limits

def compute_quality(scores):
    return float(np.mean(scores) / max(np.std(scores), 1e-12))

def compute_distance(correlation):
    correlation = np.clip(correlation, -1, 1).copy()
    correlation[np.isclose(correlation, 1, rtol=0, atol=1e-12)] = 1
    distance = np.sqrt((1-correlation)/2)
    np.fill_diagonal(distance, 0)
    return distance

def cluster_base(correlation):
    distance = compute_distance(correlation)
    count = len(distance)
    if np.max(distance) < 1e-7:
        return np.zeros(count, dtype=int), np.zeros(count)
    if count == 2:
        return np.arange(count), np.zeros(count)
    feasible = min(count-1, len(np.unique(np.round(distance,12),axis=0)))
    best_quality = -np.inf
    best_labels = None
    best_scores = None
    for seed in range(10):
        for groups in range(2,feasible+1):
            labels = KMeans(n_clusters=groups,n_init=1,random_state=seed).fit_predict(distance)
            if len(np.unique(labels)) != groups:
                continue
            scores = silhouette_samples(distance,labels)
            quality = compute_quality(scores)
            if quality > best_quality:
                best_labels, best_scores, best_quality = labels, scores, quality
    if best_labels is None:
        raise ValueError("ONC could not form a nondegenerate candidate partition")
    return best_labels, best_scores

def refine_clusters(correlation):
    labels, scores = cluster_base(correlation)
    groups = np.unique(labels)
    qualities = np.array([compute_quality(scores[labels==group]) for group in groups])
    redo = groups[qualities < qualities.mean()]
    if len(redo) <= 2:
        return labels, scores
    indices = np.flatnonzero(np.isin(labels,redo))
    refined, _ = refine_clusters(correlation[np.ix_(indices,indices)])
    candidate = np.empty(len(labels),dtype=int)
    kept = groups[~np.isin(groups,redo)]
    for group_index, group in enumerate(kept):
        candidate[labels==group] = group_index
    candidate[indices] = refined + len(kept)
    candidate_scores = silhouette_samples(compute_distance(correlation),candidate)
    candidate_quality = np.mean([compute_quality(candidate_scores[candidate==group])
                                 for group in np.unique(candidate)])
    previous_quality = qualities[np.isin(groups,redo)].mean()
    if candidate_quality > previous_quality:
        return candidate, candidate_scores
    return labels, scores

def cluster_trials(values):
    correlation = np.atleast_2d(np.corrcoef(values,rowvar=False))
    with threadpool_limits(limits=1):
        labels, scores = refine_clusters(correlation)
    quality = None if values.shape[1] <= 2 or len(np.unique(labels))==1 else compute_quality(scores)
    return labels, quality

def compute_minimum_variance_weights(covariance):
    volatility = np.sqrt(np.diag(covariance))
    if not np.isfinite(volatility).all() or (volatility<=0).any():
        raise ValueError("ONC representative requires finite positive trial variances")
    count = len(volatility)
    if count == 1:
        return np.ones(1)
    correlation = covariance / np.outer(volatility,volatility)
    average = float((correlation.sum()-count)/(count*(count-1)))
    if np.isclose(average,1,rtol=0,atol=4*np.finfo(float).eps) and np.ptp(volatility)<=4*np.finfo(float).eps*volatility.max():
        return np.full(count,1/count)
    denominator = 1+(count-1)*average
    if denominator <= 0:
        raise ValueError("ONC constant-correlation covariance is singular")
    inverse_volatility = volatility.min()/volatility
    weights = inverse_volatility**2 - average*inverse_volatility*inverse_volatility.sum()/denominator
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("ONC minimum-variance weights cannot be normalized")
    return weights/total

def compute_cluster_sharpes(values,labels):
    sharpes = []
    for group in np.unique(labels):
        members = values[:,labels==group]
        covariance = np.atleast_2d(np.cov(members,rowvar=False,ddof=1))
        weights = compute_minimum_variance_weights(covariance)
        representative = members@weights
        volatility = representative.std(ddof=1)
        if not np.isfinite(volatility) or volatility <= 1e-12*members.std(axis=0,ddof=1).max():
            raise ValueError("ONC representative has zero or numerically negligible variance")
        sharpes.append(float(representative.mean()/volatility))
    return sharpes

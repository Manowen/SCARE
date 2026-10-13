import numpy as np

def predict_delta(matrix, steps, target):
    gram = matrix @ matrix.T
    values, vectors = np.linalg.eigh(gram)
    value = max(float(values[-1]), 0.0)
    if value < 1e-20:
        return (np.zeros(matrix.shape[1], np.float32), {'energy': 0.0, 'r2': None})
    coeff = vectors[:, -1] * np.sqrt(value)
    direction = matrix.T @ vectors[:, -1] / np.sqrt(value)
    fit = np.polyfit(steps, coeff, 1)
    pred = np.polyval(fit, target)
    denom = float(np.sum((coeff - coeff.mean()) ** 2))
    return ((pred * direction).astype(np.float32), {'energy': value / max(float(np.maximum(values, 0).sum()), 1e-20), 'r2': 1 - float(np.sum((coeff - np.polyval(fit, steps)) ** 2)) / denom if denom > 1e-20 else None})

def project_delta(matrix):
    values, vectors = np.linalg.eigh(matrix @ matrix.T)
    vector = vectors[:, -1]
    return (vector[-1] * (matrix.T @ vector)).astype(np.float32)

def endpoint_delta(matrix, steps, target):
    return (matrix[-1] * (target / steps[-1])).astype(np.float32)

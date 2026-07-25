import numpy as np

def precision_at_k(actual, predicted, k=10):
    predicted_k = predicted[:k]
    return len(set(actual) & set(predicted_k)) / k

def recall_at_k(actual, predicted, k=10):
    predicted_k = predicted[:k]
    return len(set(actual) & set(predicted_k)) / len(actual) if actual else 0

def ndcg_at_k(actual, predicted, k=10):
    dcg = sum(1 / np.log2(i + 2) for i, p in enumerate(predicted[:k]) if p in actual)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(actual), k)))
    return dcg / idcg if idcg > 0 else 0

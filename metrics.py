import numpy as np 

def dcg_at_k(rels, k):
    rels = np.asarray(rels)[:k]
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return np.sum(rels * discounts)

def ndcg_at_k(y_true, y_pred, k):
    order = np.argsort(-y_pred)
    ideal = np.argsort(-y_true)
    dcg   = dcg_at_k(y_true[order], k)
    idcg  = dcg_at_k(y_true[ideal], k)
    return dcg / idcg if idcg > 0 else 0.0

def precision_recall_at_k(y_true, y_pred, k, thresh=4.0):
    idx = np.argsort(-y_pred)[:k]
    rel = (y_true[idx] >= thresh).astype(int)
    precision = rel.mean()
    recall    = rel.sum() / max((y_true >= thresh).sum(), 1)
    return precision, recall

def average_precision_at_k(y_true, y_pred, k, thresh=4.0):
    idx = np.argsort(-y_pred)[:k]
    rel = (y_true[idx] >= thresh).astype(int)
    if rel.sum() == 0:
        return 0.0
    cum = [rel[:i+1].mean() for i in range(len(rel)) if rel[i]]
    return float(np.mean(cum))

def f_measure_at_k(y_true, y_pred, k):
    precision, recall= precision_recall_at_k(y_true, y_pred, k)
    f1= 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
    return f1



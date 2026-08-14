def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    top_k = set(recommended[:k])
    set_rel = set(relevant)
    result = set_rel.intersection(top_k)
    return [len(result)/k, len(result)/len(relevant)]
    
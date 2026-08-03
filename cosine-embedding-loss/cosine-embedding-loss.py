def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    # Write code here

    # Dot product
    dot_product = sum(a*b for a, b in zip(x1, x2))

    # Norms 
    norm_x1 = math.sqrt(sum(a*a for a in x1))
    norm_x2 = math.sqrt(sum(b*b for b in x2))
    product_of_norms = norm_x1 * norm_x2

    # Cosine Similarity
    cosine_sim = dot_product / product_of_norms

    if label == 1:
        loss = 1 - cosine_sim
    elif label == -1:
        loss = max(0, cosine_sim - margin)

    return loss
    
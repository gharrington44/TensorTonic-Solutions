import math

def log_loss(y_true, y_pred, eps=1e-15):
    """
    Compute per-sample log loss.
    """
    # Clip prediction to avoid log(0)
    losses = []
    
    for yt, yp in zip(y_true, y_pred):
        yp = max(eps, min(yp, 1 - eps))
        loss = -(yt * math.log(yp) + (1 - yt) * math.log(1 - yp))
        losses.append(loss)

    return losses
    
        
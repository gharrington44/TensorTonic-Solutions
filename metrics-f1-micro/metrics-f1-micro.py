def f1_micro(y_true: list[int], y_pred: list[int]) -> float:
    """
    Return the micro-averaged F1 score rounded to four decimals.
    """
    # TP: Model predicts class and true label  = class
    # FP: Model predicted class but it was actually something else
    # FN: Sample belonged to the class but model prediced something else
    # Write code here
    TP = 0
    FN = 0
    FP = 0
    
    # Step 1: Get all the class labels: Take the set of combined y_true and y_pred
    classes = set(y_true + y_pred)

    # Step 2: Loop through the classes and then loop through the samples
    for val in classes:
        for i in range(len(y_true)):
            if val == y_pred[i] and val == y_true[i]:           # TP
                TP += 1
            elif val == y_true[i] and y_true[i] != y_pred[i]:   # FN
                FN += 1
            elif val == y_pred[i] and y_true[i] != y_pred[i]:   # FP
                FP += 1
            
    F1 = 2*TP/(2*TP + FP + FN)
    
    return round(float(F1), 4)
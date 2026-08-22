import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    Break ties by choosing the smallest class label.
    """
    predictions = np.array(predictions)

    num_trees = predictions.shape[0]
    num_samples = predictions.shape[1]

    res = []

    for sample in range(num_samples):
        class_counts = {}

        # Count votes for this sample across all trees
        for tree in range(num_trees):
            label = predictions[tree, sample]

            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1

        # Find the class with the most votes
        best_class = None
        best_count = -1

        for label, count in class_counts.items():
            if count > best_count:
                best_class = label
                best_count = count
            elif count == best_count and label < best_class:
                best_class = label

        res.append(best_class)

    return res
"""Small, explicit metrics shared by evaluation and artifact verification."""

from collections import Counter


def classification_metrics(targets, predictions, class_names):
    if len(targets) != len(predictions) or not targets:
        raise ValueError("Targets and predictions must have the same nonzero length.")
    size = len(class_names)
    if not size or len(set(class_names)) != size:
        raise ValueError("Class names must be nonempty and unique.")
    if any(type(x) is not int or not 0 <= x < size for x in [*targets, *predictions]):
        raise ValueError("Class IDs must be integers in the class mapping.")
    confusion = [[0] * size for _ in range(size)]
    for target, prediction in zip(targets, predictions):
        confusion[target][prediction] += 1
    counts = Counter(targets)
    per_class = [
        {
            "class_id": i,
            "class_name": name,
            "support": counts[i],
            "correct": confusion[i][i],
            "accuracy": confusion[i][i] / counts[i] if counts[i] else None,
        }
        for i, name in enumerate(class_names)
    ]
    correct = sum(row["correct"] for row in per_class)
    present = [row["accuracy"] for row in per_class if row["support"]]
    return {
        "num_samples": len(targets),
        "num_classes": size,
        "correct": correct,
        "top1_accuracy": correct / len(targets),
        "top1_accuracy_percent": 100 * correct / len(targets),
        "mean_per_class_accuracy": sum(present) / len(present),
        "mean_per_class_accuracy_percent": 100 * sum(present) / len(present),
        "classes_with_support": len(present),
    }, per_class, confusion

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

class Classifier:
    """
    Receives model predictions and true labels, computes evaluation metrics,
    and stores the confusion matrix.
    """

    def __init__(self, estimated_counts, true_counts, categories_name):
        self.estimated_counts  = estimated_counts
        self.evaluated_targets = true_counts
        self.categories_name   = categories_name
        self.metric_names      = ['Accuracy', 'F1-Score (macro)', 'F1-Score (weighted)']
        self._computeMetrics()

    def _computeMetrics(self):
        acc = accuracy_score(self.evaluated_targets, self.estimated_counts)
        f1_macro = f1_score(self.evaluated_targets, self.estimated_counts, average='macro', zero_division=0)
        f1_weighted = f1_score(self.evaluated_targets, self.estimated_counts, average='weighted', zero_division=0)
        self.evaluation_metrics = [acc, f1_macro, f1_weighted]
        self.confusion_mat = confusion_matrix(self.evaluated_targets, self.estimated_counts)
        print(f"\n{'='*40}")
        print(f"  Accuracy:             {acc * 100:.2f}%")
        print(f"  F1-Score (macro):     {f1_macro:.4f}")
        print(f"  F1-Score (weighted):  {f1_weighted:.4f}")
        print(f"{'='*40}\n")
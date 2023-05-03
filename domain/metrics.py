import sklearn.metrics

def get_metrics(y_true, y_pred):
    return {
        "accuracy": sklearn.metrics.accuracy_score(y_true, y_pred),
        "precision": sklearn.metrics.precision_score(y_true, y_pred),
        "recall": sklearn.metrics.recall_score(y_true, y_pred),
        "f1": sklearn.metrics.f1_score(y_true, y_pred),
        "auc": sklearn.metrics.roc_auc_score(y_true, y_pred),
        "mcc": sklearn.metrics.matthews_corrcoef(y_true, y_pred)
    }


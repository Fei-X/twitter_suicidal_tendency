import numpy as np
from sklearn import metrics

# Beta was set to 2 because we wan't every suicidal case being discoverd.
BETA = 2 
THRESHOLDS = 100

def threshold(y_pred_proba, y_test):
    global THRESHOLDS

    t_opt = 0
    f_beta_opt = 0

    for t in np.arange(0, 1, 1 / THRESHOLDS):
        y_pred = (y_pred_proba[:,1] > t).astype(int)
        f_beta = metrics.fbeta_score(
            y_test, 
            y_pred,
            beta = BETA
        )
    
        if f_beta > f_beta_opt:
            t_opt = t
            f_beta_opt = f_beta
    
    y_pred = (y_pred_proba[:,1] > t_opt).astype(int)
    return {'threshold': t_opt, 'score': f_beta_opt, 'y_pred': y_pred}

def performance(y_test, y_pred, y_pred_proba):
    report = """
The evaluation report of classification is:
Confusion Matrix:
{}
Accuracy: {}
Precision: {}
Recall: {}
F2 Score: {}
AUC Score: {}
""".format(metrics.confusion_matrix(y_test, y_pred),
           metrics.accuracy_score(y_test, y_pred),
           metrics.precision_score(y_test, y_pred),
           metrics.recall_score(y_test, y_pred),
           metrics.fbeta_score(y_test, y_pred, beta = BETA),
           metrics.roc_auc_score(y_test, y_pred_proba))
    return {
        'report': report,
        'cm': metrics.confusion_matrix(y_test, y_pred),
        'accuracy': metrics.accuracy_score(y_test, y_pred),
        'precision': metrics.precision_score(y_test, y_pred),
        'recall': metrics.recall_score(y_test, y_pred),
        'f_beta': metrics.fbeta_score(y_test, y_pred, beta = BETA),
        'AUC': metrics.roc_auc_score(y_test, y_pred)
    }


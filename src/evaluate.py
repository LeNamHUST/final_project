import numpy as np


def f1_score(y_true, y_pred):
    def recall(y_true, y_pred):
         true_positive = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
         posible_positives =  np.sum(np.round(np.clip(y_true, 0, 1)))
         recall = true_positive/(posible_positives + np.finfo(float).eps)
         return recall
    def precision(y_true, y_pred):
         true_positive = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
         predicted_positives =  np.sum(np.round(np.clip(y_pred, 0, 1)))
         precision = true_positive/(predicted_positives + np.finfo(float).eps)
         return precision
    try:
        true_np = y_true.to("cpu").numpy()
        pred_np = y_pred.to("cpu").numpy()
    except:
        true_np = y_true.to("cpu").numpy()
        pred_np = y_pred.detach().cpu().numpy()
    precision = precision(true_np, pred_np)
    recall = recall(true_np, pred_np)
    return 2*((precision*recall)/(precision+recall+np.finfo(float).eps))
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

def get_embeddings(model, anchors, positives, negatives):
    prediction = model.predict([anchors, positives, negatives], verbose=0)
    emb_a = prediction[:, :128]
    emb_p = prediction[:, 128:256]
    emb_n = prediction[:, 256:]
    return emb_a, emb_p, emb_n

def calculate_triplet_accuracy(emb_a, emb_p, emb_n):
    y_true = []; y_pred = []

    for a, p, n in zip(emb_a, emb_p, emb_n):
        d_ap = np.linalg.norm(a - p)
        d_an = np.linalg.norm(a - n)
        y_true.append(1)
        y_pred.append(1 if d_ap < d_an else 0)

    acc = np.mean([pred == true for pred, true in zip(y_pred, y_true)])
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
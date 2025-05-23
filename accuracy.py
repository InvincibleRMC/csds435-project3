from sklearn.metrics import accuracy_score
from surprise.prediction_algorithms import Prediction


def accuracy(predictions: list[Prediction], verbose: bool = True) -> float:
    y_true = []
    y_pred = []

    for _, _, true_r, est, _ in predictions:
        # Has to force a class for accuracy
        y_true.append(int(true_r))
        y_pred.append(round(est))

    acc = accuracy_score(y_true, y_pred)

    if verbose:
        print(f"ACC: {acc:1.4f}")

    return acc

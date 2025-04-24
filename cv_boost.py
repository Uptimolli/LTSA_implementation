import catboost as cb
import numpy as np
from sklearn.metrics import mean_squared_error as mse, log_loss, roc_auc_score  # noqa
from utils import get_splitter, tts_by_fold_indexes


def CV_boosting(data, target, test_data, task="binary", n_folds=5, cb_params=None):
    """
    Perform cross-validation with boosting models.

    Parameters:
    - data: pandas DataFrame with features and target
    - target: name of the target column
    - task: 'binary' or 'multiclas' for classification or 'reg' for regression
    - n_folds: number of folds for cross-validation
    - params: dictionary of parameters for the boosting model
    - verbose: whether to print progress information

    Returns:
    - models: list of trained models for each fold
    - oof_preds: out-of-fold predictions
    - cv_score: average cross-validation score
    """
    X = data.drop(columns=target)
    y = data[target]
    if cb_params is None:
        cb_params = {"verbose": 0}

    splitter = get_splitter(task, n_folds)
    if task == "binary":
        model = cb.CatBoostClassifier
        # metric = lambda x, y: log_loss(x, y)  # noqa
        metric = lambda x, y: roc_auc_score(x, y[:, 1]) if len(y.shape) == 2 else roc_auc_score(x, y)  # noqa
    else:
        model = cb.CatBoostRegressor
        metric = lambda x, y: mse(x, y) ** 0.5  # noqa

    models = []
    oof_preds = (
        np.zeros(len(X)) if task == "reg" else np.zeros((len(X), len(np.unique(y))))
    )
    fold_scores = []

    for train_idx, val_idx in splitter.split(X, y):
        fold_model = model(**cb_params)

        X_train, y_train, X_val, y_val = tts_by_fold_indexes((X, y), (train_idx, val_idx))

        fold_model.fit(
            X_train,
            y_train,
        )

        if task == "binary":
            fold_preds = fold_model.predict_proba(X_val)
            oof_preds[val_idx] = fold_preds
            fold_score = metric(y_val, fold_preds)
        else:
            fold_preds = fold_model.predict(X_val)
            oof_preds[val_idx] = fold_preds
            fold_score = metric(y_val, fold_preds)

        fold_scores.append(fold_score)
        models.append(fold_model)  # or may be copy of model?
    assert np.sum(np.array([id(model) for model in models]) == id(models[0])) == 1

    test_preds = []
    if task == "binary":
        for model in models:
            test_preds.append(model.predict_proba(test_data))
    elif task == "reg":
        for model in models:
            test_preds.append(model.predict(test_data))

    # TODO: not mean, but blend by OOF
    test_preds = np.mean(test_preds, axis=0)
    return oof_preds, test_preds, models, splitter, metric

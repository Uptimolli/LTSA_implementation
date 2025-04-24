from ASP import Adaptive_stopping_procedure

from cv_boost import CV_boosting

from utils import get_all_train_params_from_data_dict


def make_one_exp(
    dataset=None,
    n_samples=1000,
    random_state=42,
    n_iters=10,
    CV=5,
    ASP_CLUSTERS=3,
    cb_lr=0.01,
    mdl_frac=0.9,
    partition_algo_name="kmeans",
    exp_verbose=1,
    sample_frac=0.9,
):
    data, test_data, test_target, target, task, cats = (
        get_all_train_params_from_data_dict(dataset, n_samples, random_state, sample_frac=sample_frac)
    )

    cb_params = {
        "iterations": n_iters,
        "learning_rate": cb_lr,
        "verbose": 0,
    }
    ASP_PARAMS = {
        "task": task,
        "cats": cats,
        "partition_algo_name": partition_algo_name,
        "ASP_verbose": exp_verbose,
        "CV": CV,
        "CLUSTERS_NUMBER": ASP_CLUSTERS,
        "mdl_frac": mdl_frac,
        "cb_params": cb_params,
    }

    _, test_preds, models, splitter, metric = CV_boosting(
        data=data,
        target=target,
        task=task,
        test_data=test_data,
        n_folds=ASP_PARAMS["CV"],
        cb_params=cb_params,
    )

    ASP_PARAMS["fitted_models"] = models
    ASP_PARAMS["splitter"] = splitter
    ASP_PARAMS["metric"] = metric

    CV_catboost_metric = metric(test_target, test_preds)
    if exp_verbose:
        print(f"CV catboost roc auc={CV_catboost_metric}")

    asp = Adaptive_stopping_procedure(**ASP_PARAMS)
    test_preds = asp.fit_predict(data=data, target=target, test_data=test_data)

    ASP_metric = metric(test_target, test_preds)
    if exp_verbose:
        print(f"ASP roc auc={ASP_metric}")

    return ASP_metric, CV_catboost_metric

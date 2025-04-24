from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse, log_loss, roc_auc_score  # noqa
from partition_algos import get_partition_by_kmeans, get_partition_by_cluster_tree

import os

# it's for k means
default_n_threads = 8
os.environ["OPENBLAS_NUM_THREADS"] = f"{default_n_threads}"
os.environ["MKL_NUM_THREADS"] = f"{default_n_threads}"
os.environ["OMP_NUM_THREADS"] = f"{default_n_threads}"


class Adaptive_stopping_procedure:
    def __init__(
        self,
        task,
        fitted_models,
        splitter,
        metric,
        cats=None,
        CV=5,
        CLUSTERS_NUMBER=5,
        partition_algo_name="kmeans",
        mdl_frac=0.5,
        ASP_verbose=1,
        cb_params=None,
    ):
        self.CV = CV  # Cross-val split parameter
        self.CLUSTERS_NUMBER = CLUSTERS_NUMBER
        self.B = fitted_models[0].__dict__["_init_params"]["iterations"]  # num of boosting iterations
        self.task = task
        self.partition_algo_name = partition_algo_name
        self.partition_algo = None
        self.ASP_verbose = ASP_verbose

        if partition_algo_name == "cluster_tree":
            self.mdl_frac = mdl_frac
        self.fitted_models = fitted_models
        self.splitter = splitter
        self.metric = metric

        if cats is None:
            self.cats = []
        else:
            self.cats = cats

        if cb_params is None:
            self.cb_params = {"verbose": 0}
        else:
            self.cb_params = cb_params

    def _fit(self, X, y, k=None):
        if k is None:
            k = self.CV
        S = (X, y)

        # self.tts_folds_split(k, S)
        self.folds = [
            (train_idx, test_idx) for train_idx, test_idx in self.splitter.split(X, y)
        ]

        self.GetPartition(
            S, algorithm=self.partition_algo_name, n_clusters=self.CLUSTERS_NUMBER
        )

        self.bestIter = self.EvalBestIter(S, self.fitted_models)

    def GetPartition(self, S, n_clusters=None, algorithm="kmeans"):
        if n_clusters is None:
            n_clusters = self.CLUSTERS_NUMBER
        X, y = S

        if algorithm == "kmeans":
            partition, algo = get_partition_by_kmeans(X, y, n_clusters)
        elif algorithm == "cluster_tree":
            mdl = int(self.mdl_frac * y.shape[0] / self.CLUSTERS_NUMBER)
            partition, algo = get_partition_by_cluster_tree(
                X,
                y,
                n_clusters=n_clusters,
                task=self.task,
                fitted_models=self.fitted_models,
                splitter=self.splitter,
                metric=self.metric,
                mdl=mdl,
            )
        else:
            raise NotImplementedError(f"Algorithm {algorithm} not implemented")

        self.partition = partition
        self.partition_algo = algo

    def EvalBestIter(
        self,
        S,
        models,
    ):
        """
        What's the idea?

        S: Dataset S=(X, y). X is already one-hot encoded
        folds: Folds of size (CV, 2, [N / CV, N * (CV - 1) / CV]), indexing how the space is divided for cross-validation
        partition: Partition of size (CLUSTERS_NUMBER, 2, [N / CLUSTERS_NUMBER, N * (CLUSTERS_NUMBER - 1) / CLUSTERS_NUMBER]),
            this is the indexing of how the space is divided for clustering
        task: Type of task is being solved (needed to select the metric and algorithm (regression/classification))
        models: Previously trained models on cross-validation

        The question is, "how to calculate the best iteration for each subspace?":

        1. Take the i-th subspace, partition, (partitions[i]). For it, we want to determine the best iteration for early stopping
        2. Take the j-th fold. Get a new subspace obtained by intersecting the fold and partition.
        3. Take the CatBoost model that was not trained on this fold.
        4. For this CatBoost, iterate through all B_size from 1 to B, calculating the error/metric for each size.
        5. Iterate through all folds of this subspace, getting errors for each B_size.
        6. For this subspace, choose the best B_size as the one with the minimum error value.
        7. Repeat this for all partitions.
        """

        assert self.task == "binary", "Sorry, solving only binary task now"

        X, y = S
        labels = np.unique(y)

        # train and val lists in partition aren't empty
        assert all(p[0].shape[0] > 0 for p in self.partition)
        assert all(p[1].shape[0] > 0 for p in self.partition)

        best_iters = [0] * len(self.partition)

        # Also we can try to use list `history` from get_partition_by_cluster_tree, may be it can speed up algo
        # currently it's bottleneck of ASP
        for i, (_, D_i) in enumerate(self.partition):
            L_i = np.zeros(shape=self.B)

            for j, (_, S_j) in enumerate(self.folds):
                D_ij = list(set(D_i).intersection(set(S_j)))
                n_ij = len(D_ij)

                data_subspace = X.iloc[D_ij]
                ground_truth = y.iloc[D_ij]
                assert len(ground_truth) > 0

                staged_predict_probas = models[j].staged_predict_proba(
                    data_subspace, ntree_end=self.B
                )
                staged_predict_probas = np.array([i for i in staged_predict_probas])

                for k, staged_predict_proba in enumerate(staged_predict_probas):
                    if len(np.unique(ground_truth)) > 1:
                        k_value = self.Eval(
                            true=ground_truth,
                            pred=staged_predict_proba,
                            task=self.task,
                            labels=labels,
                        )
                    else:
                        print("Ground truth array is const")
                        if self.task == "binary":
                            k_value = 0.5
                        else:
                            k_value = np.inf  # this is REALLY bad, i should think to replace with big number or just skip this part

                    L_i[k] += k_value * n_ij

            # assert n_i == len(D_i)
            L_i /= len(D_i)

            if self.task == "binary":
                best_iters[i] = np.argmax(L_i)
            else:
                best_iters[i] = np.argmin(L_i)
            if self.ASP_verbose:
                print(f"[{i + 1}/{self.CLUSTERS_NUMBER}] Best iter={best_iters[i]}")

        return best_iters

    def predict(self, test_data: pd.DataFrame):
        """
        test_data: Data for prediction

        Idea:
        We have already determined which iterations are the best.
        Now we want to make predictions:
        1. For incoming data, we determine the partition for each instance.
        2. We take models (copies?) and shrink them to the best iteration.
        3. We make predictions using all models
        """

        if self.partition_algo_name == "cluster_tree":
            test_data_partition = self.partition_algo.predict(test_data)[:, -1]
        elif self.partition_algo_name == "kmeans":
            test_data_partition = self.partition_algo.predict(test_data)
        else:
            assert False, f"Unknown partition algorithm: {self.partition_algo_name}"

        if self.task == "reg":
            predictions = np.zeros(shape=len(test_data))
        else:
            n_classes = len(self.fitted_models[0].classes_)
            predictions = np.zeros(shape=(len(test_data), n_classes))

        predictions = pd.DataFrame(predictions, index=test_data.index)
        for cluster_id in np.unique(test_data_partition):
            best_iter = self.bestIter[cluster_id]

            this_cluster_indexes = test_data.loc[
                test_data_partition == cluster_id
            ].index
            model_preds = []
            for model in self.fitted_models:
                if self.task == "reg":
                    pred = model.predict(
                        test_data.loc[this_cluster_indexes], ntree_end=best_iter + 1
                    )
                else:
                    pred = model.predict_proba(
                        test_data.loc[this_cluster_indexes], ntree_end=best_iter + 1
                    )
                model_preds.append(pred)

            final_preds = np.mean(model_preds, axis=0)
            predictions.loc[this_cluster_indexes] = final_preds

        return np.array(predictions)

    @staticmethod
    def Eval(true, pred, task, labels=None):
        if task == "reg":
            loss = lambda x, y: mse(x, y) ** 0.5  # noqa
        else:  # only binary right now
            # loss = lambda x, y: log_loss(x, y, labels=labels)  # noqa
            loss = lambda x, y: roc_auc_score(x, y[:, 1])  # noqa

        return loss(true, pred)

    def TrainCVModels(self, k, S, cb_params: Optional[dict] = None):
        raise NotImplementedError(
            "This algorithm is redundant, the already fitted CatBoost models are in self.fitted_models"
        )

    def fit(self, data, target):
        self._fit(X=data.drop(columns=target), y=data[target])

    def fit_predict(self, data, target, test_data):
        self.fit(data, target)
        return self.predict(test_data)

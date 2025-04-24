from sklearn.cluster import KMeans
import numpy as np

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from py_boost.cv import ClusterCandidates
from utils import tts_by_fold_indexes


def get_partition_by_kmeans(X, y, n_clusters=5):
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)

    partition = km.fit_predict(X, y)
    partition_list = []
    for n_cluster in range(n_clusters):
        not_in_cluster_indices = np.where(partition != n_cluster)[0]
        in_cluster_indices = np.where(partition == n_cluster)[0]
        partition_list.append((not_in_cluster_indices, in_cluster_indices))
    return partition_list, km


def get_partition_by_cluster_tree(
    X, y, fitted_models, splitter, metric, task, n_clusters=5, mdl=5
):
    assert fitted_models is not None
    assert splitter is not None
    assert metric is not None

    assert task == "binary", "Sorry, only binary now"

    n_iters = fitted_models[0].__dict__["_init_params"]["iterations"]
    indices = np.arange(n_iters)

    histories = np.zeros((len(y), n_iters))

    for i, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
        _, _, X_val, y_val = tts_by_fold_indexes((X, y), (train_idx, val_idx))

        clf = fitted_models[i]
        p = np.array(list(clf.staged_predict_proba(data=X_val)))[indices, :, 1]

        for j, p0 in enumerate(p):
            logloss = -(y_val * np.log(p0) + (1 - y_val) * np.log(1 - p0))
            assert logloss.shape[0] > 1
            histories[val_idx, j] = logloss
    clustering = ClusterCandidates(depth_range=list(range(1, 10)), min_data_in_leaf=mdl)

    clustering.max_clust = n_clusters

    clustering.fit(X, histories)
    clusters_train = clustering.predict(X)

    not_const_partition = -1
    for depth in range(clusters_train.shape[1] - 1, -1, -1):
        if len(np.unique(clusters_train[:, depth])) > 1:
            not_const_partition = depth
            break
    assert not_const_partition != -1

    partition = clusters_train[:, not_const_partition]

    train_test_indices = []

    for fold in np.unique(partition):
        train_indices = np.where(partition != fold)[0]
        test_indices = np.where(partition == fold)[0]
        train_test_indices.append((train_indices, test_indices))

    return train_test_indices, clustering

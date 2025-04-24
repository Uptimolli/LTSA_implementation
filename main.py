from itertools import product
import numpy as np

from warnings import filterwarnings
from make_exp import make_one_exp
import argparse


filterwarnings("error")

parser = argparse.ArgumentParser(description='Run LTSA experiments')
parser.add_argument('--verbose', type=int, default=0, help='Verbosity level')
parser.add_argument('--n_seeds', type=int, default=5, help='Number of seeds to run')
parser.add_argument('--start_seed', type=int, default=49, help='Starting seed value')
parser.add_argument('--dataset', type=str, default="jobs", help='Dataset to use')
parser.add_argument('--CV', type=int, default=5, help='Cross-val folds number')
parser.add_argument('--partition_algo_name', "--pan", type=str, default="cluster_tree", help='Name of partition algorithm')
args = parser.parse_args()

verbose = args.verbose
N_SEEDS = args.n_seeds
START_SEED = args.start_seed
dataset = args.dataset
CV = args.CV
partition_algo_name = args.partition_algo_name

dct_of_shorts = {"ct": "cluster_tree", "km": "kmeans"}
partition_algo_name = dct_of_shorts.get(partition_algo_name, partition_algo_name)



# hyperparameters:
# 1. if dataset is None we can get data from sklearn.make_... method, sample_size is size for this method
sample_sizes = (2000,)
# 2. Num of clusters
cluster_nums = (5, 7, 17)
# 3. Iterations of catboost
iters = (100, 1000, 2500)
# 4. Learning rate of catboost
LRs = (0.03, 0.1, None)
# 5. Fraction of all rows number for "min_data_in_leaf" of "cluster_tree" partition algorithm,
# ex: rows_number=1000, cluster_min_size=0.4, n_clusters=5 => min_data_in_leaf = (1000 * 0.4) / 5 = 40
cluster_min_sizes = (0.3, 0.5, 0.8, 0.99)

if partition_algo_name == "kmeans":
    cluster_min_sizes = [1]

# all combinations of this params, instead of 5 nested `for` cycles
products = product(sample_sizes, iters, cluster_nums, LRs, cluster_min_sizes)

for i, (n_samples, n_iters, ASP_CLUSTERS, cb_lr, mdl_frac) in enumerate(products):
    try:
        assert 0 < mdl_frac <= 1

        exp_params = {
            "n_iters": n_iters,
            "CV": CV,
            "ASP_CLUSTERS": ASP_CLUSTERS,
            "cb_lr": cb_lr,
            "n_samples": n_samples,
            "dataset": dataset,
            "exp_verbose": verbose,
            "mdl_frac": mdl_frac,
            "partition_algo_name": partition_algo_name,
        }

        this_params_wins = {"ASP": [0, []], "CB": [0, []]}
        print("CLUSTERS:", ASP_CLUSTERS)
        print("iters:", n_iters)
        print("cb_Lr:", cb_lr)
        print("mdl_frac:", mdl_frac)

        for random_state in range(START_SEED, START_SEED + N_SEEDS):
            exp_params["random_state"] = random_state

            ASP_roc_auc, CV_catboost_roc_auc = make_one_exp(**exp_params)
            win_roc_auc = round(abs(ASP_roc_auc - CV_catboost_roc_auc), 4)

            this_state_winner, this_state_loser = (
                ("ASP", "CB") if ASP_roc_auc > CV_catboost_roc_auc else ("CB", "ASP")
            )

            this_params_wins[this_state_winner][0] += 1
            this_params_wins[this_state_winner][1].append(win_roc_auc)

        assert this_params_wins["ASP"][0] + this_params_wins["CB"][0] > 0
        winner, loser = ("ASP", "CB") if this_params_wins["ASP"][0] > this_params_wins["CB"][0] else ("CB", "ASP")
        print("Winner for this params is", winner)

        print("Winner median diff of metric is", round(np.median(this_params_wins[winner][1]), 4))
        if this_params_wins[loser][0] != 0:
            print("Loser median diff of metric is", round(np.median(this_params_wins[loser][1]), 4))
        
        # this_params_wins format: {'ASP': [win_count, [diff1, diff2, ...]], 'CB': [win_count, [diff1, diff2, ...]]}
        # for every algo we have how much he won and diff between metric at the seed
        print(this_params_wins)
        print()
    except Exception as e:
        print(f"Fail with {i} iteration: {e}")

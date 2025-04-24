import optuna

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

def objective(trial):
    # ASP_CLUSTERS = trial.suggest_categorical("ASP_CLUSTERS", [5, 7, 17])
    ASP_CLUSTERS = [3, 5, 7, 11, 21, 27]
    ASP_CLUSTERS_int = trial.suggest_int("ASP_CLUSTERS_int", 0, len(ASP_CLUSTERS) - 1)
    dct = dict(zip(range(len(ASP_CLUSTERS)), ASP_CLUSTERS))
    ASP_CLUSTERS = dct[ASP_CLUSTERS_int]

    # n_iters = trial.suggest_categorical("n_iters", [10, 50, 100]) #, 1000, 2500, 6000])
    n_iters = [10, 50, 100, 500]
    n_iters_int = trial.suggest_int("n_iters_int", 0, len(n_iters) - 1)
    dct = dict(zip(range(len(n_iters)), n_iters))
    n_iters = dct[n_iters_int]

    # cb_lr = trial.suggest_categorical("cb_lr", [0.001, 0.005, 0.01, 0.03, 0.1, None,])
    cb_lr = [0.0001, 0.001, 0.005, 0.01, 0.03, 0.07, 0.1, None]
    cb_lr_int = trial.suggest_int("cb_lr_int", 0, len(cb_lr) - 1)
    dct = dict(zip(range(len(cb_lr)), cb_lr))
    cb_lr = dct[cb_lr_int]

    mdl_frac = trial.suggest_float('cluster_min_sizes', 0.1, 0.99)

    try:
        exp_params = {
            "n_iters": n_iters,
            "CV": CV,
            "ASP_CLUSTERS": ASP_CLUSTERS,
            "cb_lr": cb_lr,
            "n_samples": 2000,
            "dataset": dataset,
            "exp_verbose": verbose,
            "mdl_frac": mdl_frac,
            "partition_algo_name": partition_algo_name,
        }
        
        this_params_wins = {"ASP": [0, []], "CB": [0, []]}
        
        for random_state in range(START_SEED, START_SEED + N_SEEDS):
            exp_params["random_state"] = random_state

            ASP_roc_auc, CV_catboost_roc_auc = make_one_exp(**exp_params)
            win_roc_auc = abs(ASP_roc_auc - CV_catboost_roc_auc)

            this_state_winner, _ = (
                ("ASP", "CB") if ASP_roc_auc > CV_catboost_roc_auc else ("CB", "ASP")
            )

            this_params_wins[this_state_winner][0] += 1
            this_params_wins[this_state_winner][1].append(win_roc_auc)

        assert this_params_wins["ASP"][0] + this_params_wins["CB"][0] > 0
        return round(sum(this_params_wins["ASP"][1]) - sum(this_params_wins["CB"][1]), 6)
    except: # noqa
        return -0.5

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, show_progress_bar=True, n_trials=50)
    print(f"Best params is {study.best_params} with value {study.best_value}")

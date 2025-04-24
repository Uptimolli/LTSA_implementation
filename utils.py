import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, KFold

def get_data(key=None, n_samples=1000, random_state=42):
    """
    Return: data, test_data, target, task, cats
    """
    if key is not None:
        assert key in dataset_dict, "key isn't in available available datasets"
        return dataset_dict[key]()
    return get_sklearn("binary", n_samples=n_samples, random_state=random_state)


def get_jobs():
    data = pd.read_csv("data/jobs_train.csv")

    target = "target"
    task = "binary"

    cats = [
        "city",
        "gender",
        "relevant_experience",
        "enrolled_university",
        "education_level",
        "major_discipline",
        "company_type",
    ]

    to_drop = ["enrollee_id"]
    data = data.drop(columns=to_drop)

    data, test = train_test_split(data, test_size=0.2, random_state=40)

    data_target = data[target]
    test_target = test[target]
    data = data.drop(columns=target)
    test = test.drop(columns=target)

    from sklearn.preprocessing import OneHotEncoder

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe.set_output(transform="pandas")

    data = ohe.fit_transform(data)
    test = ohe.transform(test)
    cats = []

    data[target] = data_target
    test[target] = test_target

    assert data.shape == (15326, 524)

    return data, test, target, task, cats


def get_sklearn(task_type="binary", n_samples=1000, random_state=42):
    if task_type == "reg":
        raise NotImplementedError
    elif task_type == "binary":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=500,
            n_informative=100,
            n_redundant=100,
            n_classes=2,
            random_state=random_state,
        )

        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        df["target"] = y

        task = "binary"
        target = "target"
        cats = []

        data, test = train_test_split(df, test_size=0.2, random_state=40)
    else:
        raise NotImplementedError

    return data, test, target, task, cats


def get_diabetes():
    data = pd.read_csv("data/diabetes.csv")

    target = "Outcome"
    task = "binary"
    cats = []

    to_drop = []
    data = data.drop(columns=to_drop)

    data, test = train_test_split(data, test_size=0.2, random_state=40)
    return data, test, target, task, cats


def get_housing():
    to_drop = []

    data = pd.read_csv("data/housing.csv")
    cats = []
    target = "ocean_proximity"
    task = "multiclass"

    data = data.drop(columns=to_drop)

    data, test = train_test_split(data, test_size=0.2, random_state=40)
    return data, test, target, task, cats


dataset_dict = {
    "jobs": get_jobs,
    "diabetes": get_diabetes,
    "housing": get_housing,
}


def tts_by_fold_indexes(S, folds_i):
    train_idx, val_idx = folds_i
    X, y = S
    X_train, y_train, X_val, y_val = (
        X.iloc[train_idx],
        y.iloc[train_idx],
        X.iloc[val_idx],
        y.iloc[val_idx],
    )
    return X_train, y_train, X_val, y_val


def get_splitter(task, n_folds):
    if task in ("binary", "multiclass"):
        return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        return KFold(n_splits=n_folds, shuffle=True, random_state=42)


def get_all_train_params_from_data_dict(
    dataset=None,
    n_samples=1000,
    random_state=42,
    to_sample_named_dataset=True,
    sample_frac=0.9,
):
    # correct datasets names: ('diabetes', 'cjs', 'housing', 'jobs')
    data, test, target, task, cats = get_data(
        key=dataset, n_samples=n_samples, random_state=random_state
    )

    if dataset is not None and to_sample_named_dataset:
        data = data.sample(frac=sample_frac, random_state=random_state + 100)
        test = test.sample(frac=sample_frac, random_state=random_state + 100)

    test_data, test_target = test.drop(columns=target), test[target]

    return data, test_data, test_target, target, task, cats

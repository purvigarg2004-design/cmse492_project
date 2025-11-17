# src/models/baselines.py

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def make_logistic_pipeline() -> Pipeline:
    """
    Create the tuned logistic regression pipeline used as the main baseline.

    - Standardizes all features.
    - LogisticRegression with C=3.0 and class_weight='balanced'.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            C=3.0,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
        )),
    ])
    return pipe


def make_random_forest(
    n_estimators: int = 300,
    random_state: int = 0,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """
    Create the random forest classifier used as a nonlinear benchmark.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    random_state : int
        Random seed for reproducibility.
    n_jobs : int
        Number of CPU cores to use (-1 = all).
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return rf


def make_mlp_classifier(
    hidden_layer_sizes=(16,),
    alpha: float = 0.001,
    max_iter: int = 500,
    random_state: int = 0,
) -> Pipeline:
    """
    Create the tuned neural network (MLP) model wrapped in a pipeline
    with StandardScaler.

    The defaults match the best hyperparameters found in the project:
    - one hidden layer with 16 units
    - alpha = 0.001
    """
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        max_iter=max_iter,
        random_state=random_state,
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", mlp),
    ])
    return pipe

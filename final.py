# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, style
import seaborn as sns

from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectPercentile, RFE, mutual_info_classif
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

import pickle
from pprint import pprint


style.use("ggplot")

# %% [markdown]
# # Import Data

# %%
X = pd.read_csv("../data/features.csv")
y = pd.read_csv("../data/labels.csv", header=None)

print(X.shape, y.shape)

# %% [markdown]
# ## Data formatting
#
# - `gendera` column in `X`
# - `label` as column name for `y`

# %%
X["gendera"] -= 1
y.columns = ["label"]

print(X["gendera"].unique(), y.columns)

# %% [markdown]
# ## Missing values
#
# See which features have missing values and count them

# %%
missing = X.isna().sum()
print(missing)

# %% [markdown]
# # Categorical and Numerical Column ID
#
# Which features are categorical? Which features are numerical?

# %%
cat_col_inx = [
    i for i, j in enumerate(X.columns) if set(X.loc[:, j].dropna().unique()) == {1, 0}
]
num_col_inx = list(set(range(X.shape[1])).difference(set(cat_col_inx)))

print(len(cat_col_inx), len(num_col_inx))

# %%
print(missing.iloc[num_col_inx].sum(), missing.iloc[cat_col_inx].sum())

# %% [markdown]
# *$\therefore$ There are no missing values in categorical columns*

# %% [markdown]
# # Exploratory Data Analysis
#

# %% [markdown]
# ## Correlation Heatmap for the Features

# %%
corr_matrix = X.corr()
fig, ax = plt.subplots(1, 1, figsize=(36, 28), dpi=120)
corr_heatmap = sns.heatmap(corr_matrix, annot=True, ax=ax)
corr_heatmap.get_figure().savefig(fname="../data/eda/corr_heatmap", dpi=300)

# %% [markdown]
# ## Most Correlated features

# %%
most_correlated = pd.DataFrame(
    [
        (
            i,
            corr_matrix.columns[corr_matrix.loc[i].argsort().iloc[-2]],
            corr_matrix.loc[i].sort_values().iloc[-2],
        )
        for i in corr_matrix.columns
    ],
    columns=["feature_1", "feature_2", "correlation"],
)
most_correlated.sort_values(
    by="correlation", ascending=False, ignore_index=True, inplace=True
)
print(most_correlated)

# %% [markdown]
# ## Plotting the most correlated features

# %%
fig, axes = plt.subplots(2, 4, figsize=(24, 12))
for i, (x_, y_, v) in enumerate(most_correlated.iloc[:16:2].values):
    ax = axes[i // 4, i % 4]
    ax.set_title(f"{x_} vs. {y_} [{v :.3f}]")
    X.plot(kind="scatter", x=x_, y=y_, ax=ax)

fig.savefig("../data/eda/most_corr_scatter.png", dpi=200)

# %% [markdown]
# ## Mutual Information Gain for Features
#
# In descending order of mutual information gain

# %%
mi_scores = pd.Series(
    mutual_info_classif(SimpleImputer().fit_transform(X), y.values.ravel()),
    index=X.columns,
).sort_values(ascending=False)
print(mi_scores)
mi_scores.plot(kind="bar")

# %% [markdown]
# ## Pairplot for 4 most informative features

# %%
pair4best = sns.pairplot(X.loc[:, mi_scores.index[:4]], diag_kind="kde")
pair4best.savefig("../data/eda/pair4best.png", dpi=200)

# %% [markdown]
# # Preprocessing
#
# ## Imputation
#
# - SimpleImputer -> mean, median
# - KNNImputer
#
# ## Scaling
# - Standard scaling
# - Robust scaling
# - OneHotEncoding on categorical columns
#
# ## Feature selection
#
# - RFE with decision tree
# - SelectPercentile with mutual info gain
# - PCA with 75%ile
#
# ## Over sampling
#
# - SMOTE
# - ADASYN

# %%
imputers = [
    SimpleImputer(strategy="mean"),
    SimpleImputer(strategy="median"),
    KNNImputer(n_neighbors=5),
]

feature_selectors = [
    "passthrough",
    *[RFE(estimator=DecisionTreeClassifier(), n_features_to_select=i) for i in (4,)],
    *[SelectPercentile(score_func=mutual_info_classif, percentile=i) for i in (75,)],
    PCA(n_components=12),
]

scalers = [
    "passthrough",
    StandardScaler(),
    RobustScaler(),
]


over_samplers = [
    SMOTE(random_state=42),
    ADASYN(random_state=42),
    RandomOverSampler(random_state=42),
]

# %% [markdown]
# ## Classifiers
#
# - Decision Tree
# - Random Forest
# - Naive Bayes
# - Logistic Regression
# - Support Vector Machine
# - Adaptive Boosting (AdaBoost)
# - Extreme Gradient Boosting (XGBoost)
# - Light Gradient Boosting (LightGBM)


# %%
def grid_search_classifier(
    classifier,
    X: pd.DataFrame,
    y: pd.DataFrame,
    param_grid: dict[str, list],
    verbose: int = 1,
    cv: int = 5,
    log_name: str = None,
):
    """
    Performs grid search on classifier, evaluating
    with stratified k folds on cv folds. Uses the param_grid
    for hyperparameter tuning. Stores the results in log_name
    file.
    """
    p = dict()
    for key in param_grid.keys():
        p[f"classifier__{key}"] = param_grid[key]
    param_grid = p
    print(param_grid)
    clf = Pipeline(
        steps=[
            ("imputer", None),
            (
                "feature_transformer",
                ColumnTransformer(
                    transformers=[
                        ("num", None, num_col_inx),
                        ("cat", OneHotEncoder(), cat_col_inx),
                    ],
                    remainder="passthrough",
                ),
            ),
            ("feature_selector", None),
            ("over_sampler", None),
            ("classifier", classifier),
        ]
    )

    clf.set_params(classifier=classifier)
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=dict(
            imputer=imputers,
            over_sampler=over_samplers,
            feature_transformer__num=scalers,
            feature_selector=feature_selectors,
            **param_grid,
        ),
        cv=cv,
        scoring="f1_macro",
        verbose=verbose,
        n_jobs=-1,
    )
    grid_search.fit(X, y.values.ravel())

    # Store results
    pd.DataFrame(grid_search.cv_results_).drop(["params"], axis=1).to_csv(
        f"../data/grid_search_res/{log_name}.csv", index=None
    )

    # Store the best estimator
    with open(f"../data/grid_search_res/{log_name}.pkl", "w") as f:
        pickle.dump(grid_search.best_estimator_, file=f)

    return grid_search


# %% [markdown]
# # Training Different Models

# %%
# Decision Tree Classifier
dtree_grid_search = grid_search_classifier(
    classifier=DecisionTreeClassifier(),
    X=X,
    y=y,
    log_name="decision_tree",
    param_grid=dict(
        criterion=["gini", "entropy"],
        max_depth=[None, *range(3, 20)],
        min_samples_split=[2, 5, 10, 20, 30, 40],
    ),
)

print(dtree_grid_search.best_score_)

# %%
# Random Forest
rforest_grid_search = grid_search_classifier(
    classifier=RandomForestClassifier(n_jobs=-1),
    X=X,
    y=y,
    log_name="random_forest",
    verbose=2,
    param_grid=dict(
        criterion=["gini", "entropy"],
        n_estimators=[200, 500],
        ccp_alpha=[1e-5, 1e-4],
        max_depth=[None, 5, 10, 15],
        max_features=["sqrt", "log2", None],
    ),
)

print(rforest_grid_search.best_score_)

# %%
ada_grid_search = grid_search_classifier(
    classifier=AdaBoostClassifier(),
    X=X,
    y=y,
    log_name="adaboost",
    verbose=2,
    param_grid=dict(
        n_estimators=[50, 100, 200, 500],
        learning_rate=[0.01, 0.1, 1],
        estimator=[DecisionTreeClassifier(), RandomForestClassifier()],
        estimator__criterion=["gini", "entropy", "log_loss"],
        estimator__max_features=["sqrt", "log2", None],
        algorithm=["SAMME", "SAMME.R"],
    ),
)

print(ada_grid_search.best_score_)

# %%
# XGBoost
xgb_grid_search = grid_search_classifier(
    classifier=XGBClassifier(),
    X=X,
    y=y,
    log_name="xgboost",
    param_grid=dict(
        learning_rate=[0.01, 0.05, 0.1, 0.2, 0.3],
        n_estimators=[50, 100, 200, 300, 400],
        max_depth=[3, 5, 7, 9],
    ),
)

print(xgb_grid_search.best_score_)

# %%
# Logistic regression
logreg_grid_search = grid_search_classifier(
    classifier=LogisticRegression(n_jobs=-1),
    X=X,
    y=y,
    log_name="logistic",
    param_grid=dict(
        solver=["liblinear", "saga", "lbfgs"],
        penalty=["l1", "l2", "elastic-net"],
        C=[0.1, 0.5, 1.0, 2.0],
    ),
)

print(logreg_grid_search.best_score_)

# %%
knn_grid_search = grid_search_classifier(
    classifier=KNeighborsClassifier(n_jobs=-1),
    X=X,
    y=y,
    log_name="knn",
    verbose=1,
    param_grid=dict(
        n_neighbors=np.arange(3, 30, 2),
        weights=["uniform", "distance"],
        algorithm=["auto", "ball_tree", "kd_tree", "brute"],
    ),
)

print(knn_grid_search.best_score_)

# %%
# Multi Layer Perceptron
mlp_grid_search = grid_search_classifier(
    classifier=MLPClassifier(max_iter=200),
    X=X,
    y=y,
    verbose=1,
    log_name="neural_net",
    param_grid=dict(
        hidden_layer_sizes=[(10, 20), (100,), (32, 64, 32)],
        learning_rate=["invscaling", "adaptive"],
        solver=["sgd", "adam"],
        alpha=[0.03, 0.001, 0.0002],
        learning_rate_init=[10**-i for i in range(5)],
    ),
)

print(mlp_grid_search.best_score_)

# %%
# Support vector machine
svc_grid_search = grid_search_classifier(
    classifier=SVC(),
    X=X,
    y=y,
    log_name="svc",
    param_grid=dict(
        C=[0.1, 1, 10, 100],
        kernel=["linear", "rbf", "poly"],
        gamma=["scale", "auto", 0.1, 1],
        degree=[2, 3, 4, 5],
    ),
)

print(svc_grid_search.best_score_)

# %%
pprint(
    dict(
        list(
            zip(
                "svc mlp knn logreg xgb ada rforest dtree".split(" "),
                [
                    (
                        dict(
                            list(
                                zip(
                                    "precision recall f1_score support".split(),
                                    precision_recall_fscore_support(
                                        y.values.ravel(),
                                        gs.best_estimator_.predict(X, y),
                                        average="macro",
                                    ),
                                )
                            )
                        ),
                        confusion_matrix(
                            y.values.ravel(), gs.best_estimator_.predict(X, y)
                        ),
                    )
                    for gs in [
                        svc_grid_search,
                        mlp_grid_search,
                        knn_grid_search,
                        logreg_grid_search,
                        xgb_grid_search,
                        ada_grid_search,
                        rforest_grid_search,
                        dtree_grid_search,
                    ]
                ],
            )
        )
    )
)


X_imp = KNNImputer(n_neighbors=5).fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_imp, y.values.ravel(), test_size=0.33, stratify=y
)

X_train, y_train = RandomOverSampler().fit_resample(X_train, y_train)

clf = RandomForestClassifier(
    n_estimators=500, max_features=None, n_jobs=-1, ccp_alpha=0.0001, criterion="gini"
)
clf.fit(X_train, y_train)
yhat = clf.predict(X_test)
print(classification_report(y_test, yhat), confusion_matrix(y_test, yhat))


# =======================================================================================
# Evaluating models with best config according to grid search to get the
# conf matrix, (precision, recall, f1_score) --> macro avg
# X_imp = SimpleImputer(strategy="median").fit_transform(X)
# X_imp = SimpleImputer(strategy="mean").fit_transform(X)
X_imp = KNNImputer(n_neighbors=5).fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_imp, y.values.ravel(), test_size=0.33, stratify=y
)

# X_train, y_train = ADASYN().fit_resample(X_train, y_train)
X_train, y_train = SMOTE().fit_resample(X_train, y_train)
# X_train[:, num_col_inx] = RobustScaler().fit_transform(X_train[:, num_col_inx])
# X_train[:, num_col_inx] = StandardScaler().fit_transform(X_train[:, num_col_inx])

# clf = LogisticRegression(
#     solver='newton-cholesky',
#     penalty='l2',
#     C=0.001,
#     n_jobs=-1,
#     max_iter=500,
# )

# clf = SVC(C=0.001, kernel="linear")

# clf = RandomForestClassifier(
#     n_estimators=200,
#     criterion="entropy",
#     # max_depth=None,
#     max_features=None,
#     ccp_alpha=0.01
# )

# clf = MLPClassifier(
#     activation="logistic",
#     solver="adam",
#     learning_rate="adaptive",
#     learning_rate_init=0.03,
#     hidden_layer_sizes=(20, 20),
# )

# clf = KNeighborsClassifier(n_jobs=-1, n_neighbors=100)

# clf = AdaBoostClassifier(
#     estimator=RandomForestClassifier(n_estimators=30, max_depth=3, criterion="entropy"),
#     learning_rate=0.2,
#     algorithm="SAMME.R",
#     n_estimators=50
# )

clf = DecisionTreeClassifier(
    criterion="gini",
    min_samples_split=20,
    ccp_alpha=0.0001,
    max_depth=8,
)

# xgb_params = {
#     "learning_rate": 0.02,  # Typical values: [0.01, 0.1, 0.2]
#     "n_estimators": 150,  # Number of boosting rounds
#     "max_depth": 3,  # Typical values: [3, 5, 7, 9]
#     "min_child_weight": 1,  # Typical values: [1, 3, 5]
#     "subsample": 0.8,  # Fraction of samples used for each boosting round
#     "colsample_bytree": 0.8,  # Fraction of features used for each boosting round
#     "gamma": 0,  # Minimum loss reduction required to make a further partition on a leaf node
#     "reg_alpha": 0,  # L1 regularization term on weights
#     "reg_lambda": 1,  # L2 regularization term on weights
#     "scale_pos_weight": 1,  # Controls the balance of positive and negative weights
#     "objective": "binary:logistic",  # Objective function for binary classification
#     "eval_metric": "logloss",  # Evaluation metric to be used during training
#     "seed": 42,  # Random seed for reproducibility
# }

# clf = XGBClassifier(**xgb_params)

clf.fit(X_train, y_train)
yhat = clf.predict(X_test)
print(classification_report(y_test, yhat), confusion_matrix(y_test, yhat))

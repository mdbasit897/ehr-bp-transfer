"""Model zoo: simple baselines, individual learners and the submitted ensemble (R2.4).

All preprocessing (imputation, scaling) lives inside the estimator so it is fitted on
training folds only and re-used unchanged on external data.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor,
                              StackingClassifier, StackingRegressor)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

REGRESSION_MODELS = ['mean', 'persistence', 'ridge', 'gb', 'rf', 'xgb', 'ensemble']
CLASSIFICATION_MODELS = ['mean', 'persistence', 'ridge', 'gb', 'rf', 'xgb', 'ensemble']
MODEL_LABELS = {
    'mean': 'Mean / prevalence', 'persistence': 'Persistence (last-window BP)',
    'ridge': 'Ridge / logistic', 'gb': 'Gradient boosting', 'rf': 'Random forest',
    'xgb': 'XGBoost', 'ensemble': 'Stacked ensemble (submitted)',
}


def make_imputer(kind='median', seed=42):
    if kind == 'median':
        return SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
    if kind == 'knn':
        return KNNImputer(n_neighbors=5, add_indicator=True, keep_empty_features=True)
    if kind == 'mice':
        return IterativeImputer(max_iter=20, random_state=seed, add_indicator=True,
                                keep_empty_features=True)
    raise ValueError(kind)


class ColumnSelector(BaseEstimator):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]


class PersistenceRegressor(BaseEstimator, RegressorMixin):
    """Predict the target window's mean BP as the observation window's mean BP."""

    def __init__(self, column):
        self.column = column

    def fit(self, X, y, sample_weight=None):
        self.fallback_ = float(np.average(y, weights=sample_weight))
        return self

    def predict(self, X):
        return X[self.column].fillna(self.fallback_).to_numpy(dtype=float)


class PaperEnsembleRegressor(BaseEstimator, RegressorMixin):
    """The submitted model: 0.5 * GB + 0.5 * Stacking(GB, RF, XGB -> Ridge), with
    imputation and robust scaling inside."""

    def __init__(self, random_state=42, n_jobs=-1, imputer='median'):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.imputer = imputer

    def _base(self):
        rs = self.random_state
        return [('gb', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                                 max_depth=5, random_state=rs)),
                ('rf', RandomForestRegressor(n_estimators=100, max_depth=10,
                                             random_state=rs, n_jobs=self.n_jobs)),
                ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5,
                                     random_state=rs, n_jobs=self.n_jobs))]

    def fit(self, X, y, sample_weight=None):
        self.prep_ = Pipeline([('impute', make_imputer(self.imputer, self.random_state)),
                               ('scale', RobustScaler())])
        Xs = self.prep_.fit_transform(X)
        self.gb_ = clone(self._base()[0][1]).fit(Xs, y, sample_weight=sample_weight)
        self.stack_ = StackingRegressor(self._base(), final_estimator=Ridge(alpha=1.0), cv=5)
        self.stack_.fit(Xs, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        Xs = self.prep_.transform(X)
        return 0.5 * self.gb_.predict(Xs) + 0.5 * self.stack_.predict(Xs)


class PaperEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Classification analogue of the submitted ensemble (probabilities averaged)."""

    def __init__(self, random_state=42, n_jobs=-1, imputer='median'):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.imputer = imputer

    def _base(self):
        rs = self.random_state
        return [('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                  max_depth=5, random_state=rs)),
                ('rf', RandomForestClassifier(n_estimators=100, max_depth=10,
                                              random_state=rs, n_jobs=self.n_jobs)),
                ('xgb', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,
                                      random_state=rs, n_jobs=self.n_jobs,
                                      eval_metric='logloss'))]

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.array([0.0, 1.0])
        self.prep_ = Pipeline([('impute', make_imputer(self.imputer, self.random_state)),
                               ('scale', RobustScaler())])
        Xs = self.prep_.fit_transform(X)
        self.gb_ = clone(self._base()[0][1]).fit(Xs, y, sample_weight=sample_weight)
        self.stack_ = StackingClassifier(self._base(),
                                         final_estimator=LogisticRegression(max_iter=2000),
                                         cv=5, stack_method='predict_proba')
        self.stack_.fit(Xs, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        Xs = self.prep_.transform(X)
        p = 0.5 * self.gb_.predict_proba(Xs)[:, 1] + 0.5 * self.stack_.predict_proba(Xs)[:, 1]
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(float)


def make_model(name, task, target=None, seed=42, n_jobs=-1, imputer='median'):
    imp = make_imputer(imputer, seed)
    if task == 'regression':
        if name == 'mean':
            return DummyRegressor(strategy='mean')
        if name == 'persistence':
            return PersistenceRegressor(column=f'hist_{target}_mean')
        if name == 'ridge':
            return Pipeline([('impute', imp), ('scale', StandardScaler()),
                             ('model', RidgeCV(alphas=np.logspace(-3, 3, 13)))])
        if name == 'gb':
            return Pipeline([('impute', imp), ('model', GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=seed))])
        if name == 'rf':
            return Pipeline([('impute', imp), ('model', RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=seed, n_jobs=n_jobs))])
        if name == 'xgb':
            return Pipeline([('impute', imp), ('model', XGBRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=seed,
                n_jobs=n_jobs))])
        if name == 'ensemble':
            return PaperEnsembleRegressor(random_state=seed, n_jobs=n_jobs, imputer=imputer)
    else:
        if name == 'mean':
            return DummyClassifier(strategy='prior')
        if name == 'persistence':
            return Pipeline([('select', ColumnSelector(['hist_mbp_min', 'hist_mbp_last'])),
                             ('impute', make_imputer('median')), ('scale', StandardScaler()),
                             ('model', LogisticRegression(max_iter=2000))])
        if name == 'ridge':
            return Pipeline([('impute', imp), ('scale', StandardScaler()),
                             ('model', LogisticRegression(max_iter=2000))])
        if name == 'gb':
            return Pipeline([('impute', imp), ('model', GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=seed))])
        if name == 'rf':
            return Pipeline([('impute', imp), ('model', RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=seed, n_jobs=n_jobs))])
        if name == 'xgb':
            return Pipeline([('impute', imp), ('model', XGBClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=seed,
                n_jobs=n_jobs, eval_metric='logloss'))])
        if name == 'ensemble':
            return PaperEnsembleClassifier(random_state=seed, n_jobs=n_jobs, imputer=imputer)
    raise ValueError(f'unknown model {name} for {task}')


def fit(model, X, y, sample_weight=None):
    if sample_weight is None:
        return model.fit(X, y)
    if isinstance(model, Pipeline):
        return model.fit(X, y, **{f'{model.steps[-1][0]}__sample_weight': sample_weight})
    return model.fit(X, y, sample_weight=sample_weight)


def predict(model, X, task):
    if task == 'classification':
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def model_needs_history(name):
    return name == 'persistence'

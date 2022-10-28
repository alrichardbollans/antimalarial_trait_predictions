import os
from typing import List

import numpy as np
import pandas as pd
import xgboost
from matplotlib import pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, fbeta_score, RocCurveDisplay, auc, \
    PrecisionRecallDisplay, precision_recall_curve, average_precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import make_scorer
from sklearn.svm import SVC

beta = 0.5


def get_fbeta_score(y_test, y_pred):
    out = fbeta_score(y_test, y_pred, beta=beta)
    return out


def get_clf_scores_in_run(clf, clf_acc_list, clf_train_acc_list, clf_precision_list, clf_train_precision_list,
                          transformed_train_data, y_train,
                          transformed_test_data, y_test):
    from sklearn.metrics import accuracy_score, precision_score
    try:
        clf.fit(transformed_train_data, y_train)
    except AttributeError:
        print(f'No fit method for {clf}')

    finally:
        test_prediction = clf.predict(transformed_test_data)
        train_prediction = clf.predict(transformed_train_data)
        clf_acc_list.append(accuracy_score(y_test, test_prediction))
        clf_train_acc_list.append(accuracy_score(y_train, train_prediction))

        clf_precision_list.append(precision_score(y_test, test_prediction))
        clf_train_precision_list.append(precision_score(y_train, train_prediction))

        return clf_acc_list, clf_train_acc_list, clf_precision_list, clf_train_precision_list


def get_clf_accuracy_in_run(clf, clf_score_list, clf_train_score_list, transformed_train_data, y_train,
                            transformed_test_data, y_test):
    from sklearn.metrics import accuracy_score
    clf.fit(transformed_train_data, y_train)
    clf_score_list.append(accuracy_score(y_test, clf.predict(transformed_test_data)))
    clf_train_score_list.append(accuracy_score(y_train, clf.predict(transformed_train_data)))

    return clf_score_list, clf_train_score_list


def get_clf_precision_in_run(clf, clf_score_list, clf_train_score_list, transformed_train_data, y_train,
                             transformed_test_data, y_test):
    from sklearn.metrics import precision_score
    clf.fit(transformed_train_data, y_train)
    clf_score_list.append(precision_score(y_test, clf.predict(transformed_test_data)))
    clf_train_score_list.append(precision_score(y_train, clf.predict(transformed_train_data)))

    return clf_score_list, clf_train_score_list

def output_boxplot(df: pd.DataFrame, out_file: str, y_title: str):
    scores_w_model = []
    for col in df.columns.tolist():
        if 'Unnamed' not in col:
            for val in df[col].values:
                scores_w_model.append([val, col])
    boxplot_df = pd.DataFrame(scores_w_model, columns=[y_title, 'Model'])
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.boxplot(x='Model', y=y_title, data=boxplot_df)
    plt.savefig(out_file)
    plt.close()


class FeatureModel:
    feature_model = True

    def __init__(self, name: str, feature_list: List[str]):
        self.name = name
        self.feature_list = feature_list
        self.accuracies = []
        self.precisions = []
        self.fones = []
        self.fbetas = []

    def add_cv_scores(self, X_test, y_test, test_weights=None):
        y_pred = self.predict(X_test)
        if test_weights is None:
            self.accuracies.append(accuracy_score(y_test, y_pred))
            self.precisions.append(
                precision_score(y_test, y_pred))
            self.fones.append(f1_score(y_test, y_pred))
            self.fbetas.append(fbeta_score(y_test, y_pred, beta=beta))
        else:
            self.accuracies.append(accuracy_score(y_test, y_pred, sample_weight=test_weights))
            self.precisions.append(
                precision_score(y_test, y_pred, sample_weight=test_weights))
            self.fones.append(f1_score(y_test, y_pred, sample_weight=test_weights))
            self.fbetas.append(fbeta_score(y_test, y_pred, sample_weight=test_weights, beta=beta))

    def fit(self, X, y):
        pass

    def predict(self, test_data):
        return test_data[self.feature_list].max(axis=1).tolist()


class clf_scores:
    feature_model = False
    self_training = False

    def __init__(self, name, clf_class, use_missing_vals=False, grid_search_param_grid=None, init_kwargs=None):
        self.name = name
        self.clf_class = clf_class
        self.use_missing_vals = use_missing_vals
        self.grid_search_param_grid = grid_search_param_grid
        self.accuracies = []
        self.y_reals = []
        self.predict_probas = []
        self.test_weights = []
        self.precisions = []
        self.fones = []
        self.fbetas = []
        self.init_kwargs = init_kwargs

    def do_grid_search(self, model, X_train, y_train, cv=10, all_train_weight=None, calibrate=False):
        # TODO: Imporvements: using weights in scorer

        gs = GridSearchCV(
            estimator=model,
            param_grid=self.grid_search_param_grid,
            cv=cv,
            n_jobs=-1,
            scoring=make_scorer(get_fbeta_score),
            verbose=1,
            error_score='raise'

        )
        print(model)
        if all_train_weight is not None:
            if calibrate:
                clf_sigmoid = CalibratedClassifierCV(gs, method="sigmoid")
                fitted_model = clf_sigmoid.fit(X_train, y_train, sample_weight=all_train_weight)
            else:
                fitted_model = gs.fit(X_train, y_train, sample_weight=all_train_weight)
        else:
            if calibrate:
                clf_sigmoid = CalibratedClassifierCV(gs, method="sigmoid")
                fitted_model = clf_sigmoid.fit(X_train, y_train)
            else:
                fitted_model = gs.fit(X_train, y_train)

        return fitted_model

    def add_cv_scores(self, X_train, y_train, X_test, y_test, train_weights=None, test_weights=None):
        if self.init_kwargs is not None:
            clf_instance = self.clf_class(**self.init_kwargs)
        else:
            clf_instance = self.clf_class()

        if self.grid_search_param_grid is not None:
            clf_instance = self.do_grid_search(clf_instance, X_train, y_train, all_train_weight=train_weights)
        else:
            clf_instance.fit(X_train, y_train, sample_weight=train_weights)

        y_pred = clf_instance.predict(X_test)
        y_proba = clf_instance.predict_proba(X_test)[:, 1]
        self.y_reals.append(y_test)
        self.predict_probas.append(y_proba)

        if test_weights is None:
            self.accuracies.append(accuracy_score(y_test, y_pred))
            self.precisions.append(
                precision_score(y_test, y_pred))
            self.fones.append(f1_score(y_test, y_pred))
            self.fbetas.append(fbeta_score(y_test, y_pred, beta=beta))

        else:
            self.accuracies.append(accuracy_score(y_test, y_pred, sample_weight=test_weights))
            self.precisions.append(
                precision_score(y_test, y_pred, sample_weight=test_weights))
            self.fones.append(f1_score(y_test, y_pred, sample_weight=test_weights))
            self.fbetas.append(fbeta_score(y_test, y_pred, sample_weight=test_weights, beta=beta))
            self.test_weights.append(test_weights)

        try:
            if self.grid_search_param_grid is not None:
                self.feature_importance = clf_instance.best_estimator_.feature_importances_
            else:
                self.feature_importance = clf_instance.feature_importances_
        except AttributeError as e:
            print(e)
            try:
                self.feature_importance = clf_instance.coef_
            except AttributeError as e:
                print(self.name)
                print(e)

    def predict_on_unlabelled_data(self, X_train, y_train, unlab_data, train_weights=None):

        if self.init_kwargs is not None:
            clf_instance = self.clf_class(**self.init_kwargs)
        else:
            clf_instance = self.clf_class()

        if self.grid_search_param_grid is not None:
            clf_instance = self.do_grid_search(clf_instance, X_train, y_train, all_train_weight=train_weights,
                                               calibrate=False)
        else:
            if train_weights is not None:
                clf_instance = clf_instance.fit(X_train, y_train, sample_weight=train_weights)
            else:
                clf_instance = clf_instance.fit(X_train, y_train)

        y_pred = clf_instance.predict(unlab_data)
        y_proba = clf_instance.predict_proba(unlab_data)

        return y_pred, y_proba


# class self_train_clf_scores(clf_scores):
#     feature_model = False
#     self_training = True
#
#     def add_cv_scores(self, X_train, y_train, X_test, y_test, train_weights=None, test_weights=None):
#
#         clf_instance = self.clf_class(**self.init_kwargs)
#         self_train_clf = SelfTrainingClassifier(clf_instance, k_best=1000, criterion='k_best')
#
#         self_train_clf.fit(X_train, y_train)
#
#         y_pred = self_train_clf.predict(X_test)
#         if test_weights is None:
#             self.accuracies.append(accuracy_score(y_test, y_pred))
#             self.precisions.append(
#                 precision_score(y_test, y_pred))
#             self.fones.append(f1_score(y_test, y_pred))
#             self.fbetas.append(fbeta_score(y_test, y_pred, beta=beta))
#         else:
#             self.accuracies.append(accuracy_score(y_test, y_pred, sample_weight=test_weights))
#             self.precisions.append(
#                 precision_score(y_test, y_pred, sample_weight=test_weights))
#             self.fones.append(f1_score(y_test, y_pred, sample_weight=test_weights))
#             self.fbetas.append(fbeta_score(y_test, y_pred, sample_weight=test_weights, beta=beta))


def output_scores(models: List[clf_scores], output_dir: str, filetag: str):
    acc_dict = {}
    prec_dict = {}
    fone_dict = {}
    fbeta_dict = {}

    for model in models:
        acc_dict[model.name] = model.accuracies

        prec_dict[model.name] = model.precisions
        fone_dict[model.name] = model.fones
        fbeta_dict[model.name] = model.fbetas

    acc_df = pd.DataFrame(acc_dict)
    acc_df.describe().to_csv(os.path.join(output_dir, filetag + 'acc_means.csv'))
    acc_df.to_csv(os.path.join(output_dir, filetag + 'acc.csv'))
    output_boxplot(acc_df, os.path.join(output_dir, filetag + 'accuracy_boxplot.png'), y_title='Accuracy')

    prec_df = pd.DataFrame(prec_dict)
    prec_df.describe().to_csv(os.path.join(output_dir, filetag + 'prec_means.csv'))
    prec_df.to_csv(os.path.join(output_dir, filetag + 'prec.csv'))
    output_boxplot(prec_df, os.path.join(output_dir, filetag + 'precision_boxplot.png'),
                   y_title='Precision')

    f1_df = pd.DataFrame(fone_dict)
    f1_df.describe().to_csv(os.path.join(output_dir, filetag + 'f1_means.csv'))
    f1_df.to_csv(os.path.join(output_dir, filetag + 'f1.csv'))
    output_boxplot(f1_df, os.path.join(output_dir, filetag + 'f1_boxplot.png'),
                   y_title='F1 Score')

    fbeta_df = pd.DataFrame(fbeta_dict)
    fbeta_df.describe().to_csv(os.path.join(output_dir, filetag + 'fb_means.csv'))
    fbeta_df.to_csv(os.path.join(output_dir, filetag + 'fb.csv'))
    output_boxplot(fbeta_df, os.path.join(output_dir, filetag + 'fb_boxplot.png'),
                   y_title='F0.5 Score')

    # Plot ROC Curves
    for model in models:
        if not model.feature_model:
            all_y_real = np.concatenate(model.y_reals)
            all_y_proba = np.concatenate(model.predict_probas)

            if len(model.test_weights) > 0:
                all_test_weights = np.concatenate(model.test_weights)
                precision, recall, thresholds = precision_recall_curve(all_y_real, all_y_proba,
                                                                       sample_weight=all_test_weights)
                all_ap_score = average_precision_score(all_y_real, all_y_proba, sample_weight=all_test_weights)
            else:
                precision, recall, thresholds = precision_recall_curve(all_y_real, all_y_proba)
                all_ap_score = average_precision_score(all_y_real, all_y_proba)
            plt.plot(recall, precision,
                     label=model.name + r': (AP = %0.2f)' % all_ap_score,
                     lw=2, alpha=.8)
            curve_df = pd.DataFrame(
                {'precision': list(precision), 'recall': list(recall), 'thresholds': list(thresholds) + [1]})
            curve_df.to_csv(os.path.join(output_dir, '_'.join([model.name, 'pr_curves.csv'])))

            above_eighty_threshold = curve_df[(curve_df['precision'] >= 0.8) & (curve_df['precision'] >= 0.2)]
            above_eighty_threshold.to_csv(
                os.path.join(output_dir, '_'.join([model.name, 'above_eighty_precisions.csv'])))

            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(output_dir, '_'.join([filetag, 'pr_curves.png'])))
            plt.close()


def output_feature_importance(models: List[clf_scores], output_dir: str, filetag: str):
    for model in models:
        acc_df = pd.DataFrame(model.feature_importance)
        acc_df.to_csv(os.path.join(output_dir, filetag + model.name + '_feature_importance.csv'))


class LogisticRegressionWithThreshold(LogisticRegression):

    def __init__(self, penalty="l2",
                 *,
                 dual=False,
                 tol=1e-4,
                 C=1.0,
                 fit_intercept=True,
                 intercept_scaling=1,
                 class_weight=None,
                 random_state=None,
                 solver="lbfgs",
                 max_iter=100,
                 multi_class="auto",
                 verbose=0,
                 warm_start=False,
                 n_jobs=None,
                 l1_ratio=None, threshold=0.5):

        super().__init__(penalty=penalty,

                         dual=dual,
                         tol=tol,
                         C=C,
                         fit_intercept=fit_intercept,
                         intercept_scaling=intercept_scaling,
                         class_weight=class_weight,
                         random_state=random_state,
                         solver=solver,
                         max_iter=max_iter,
                         multi_class=multi_class,
                         verbose=verbose,
                         warm_start=warm_start,
                         n_jobs=n_jobs,
                         l1_ratio=l1_ratio)
        self.threshold = threshold

    def predict(self, X):
        if self.threshold == 0.5:  # If normal threshold passed in, simply call the base class predict
            return LogisticRegression.predict(self, X)
        else:
            y_scores = LogisticRegression.predict_proba(self, X)[:, 1]
            y_pred_with_threshold = (y_scores >= self.threshold).astype(int)

            return y_pred_with_threshold

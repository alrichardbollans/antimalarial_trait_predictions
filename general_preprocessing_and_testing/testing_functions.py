import os
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score, fbeta_score, precision_recall_curve, \
    average_precision_score, recall_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

beta = 0.5


def get_fbeta_score(y_test, y_pred):
    out = fbeta_score(y_test, y_pred, beta=beta)
    return out


def output_boxplot(df: pd.DataFrame, out_file: str, y_title: str):
    scores_w_model = []
    for col in df.columns.tolist():
        if 'Unnamed' not in col:
            for val in df[col].values:
                scores_w_model.append([val, col])
    boxplot_df = pd.DataFrame(scores_w_model, columns=[y_title, 'Model'])
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rc('font', size=13)
    sns.boxplot(x='Model', y=y_title, data=boxplot_df)
    plt.tight_layout()
    plt.savefig(out_file, dpi=400)
    plt.close()
    plt.cla()
    plt.clf()


class FeatureModel:
    feature_model = True

    def __init__(self, name: str, feature_list: List[str]):
        self.name = name
        self.feature_list = feature_list
        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.fones = []
        self.fbetas = []

    def add_cv_scores(self, X_test, y_test, test_weights=None):
        y_pred = self.predict(X_test)
        if test_weights is None:
            self.accuracies.append(accuracy_score(y_test, y_pred))
            self.precisions.append(
                precision_score(y_test, y_pred))
            self.recalls.append(recall_score(y_test, y_pred))
            self.fones.append(f1_score(y_test, y_pred))
            self.fbetas.append(fbeta_score(y_test, y_pred, beta=beta))
        else:
            self.accuracies.append(accuracy_score(y_test, y_pred, sample_weight=test_weights))
            self.precisions.append(
                precision_score(y_test, y_pred, sample_weight=test_weights))
            self.recalls.append(recall_score(y_test, y_pred, sample_weight=test_weights))
            self.fones.append(f1_score(y_test, y_pred, sample_weight=test_weights))
            self.fbetas.append(fbeta_score(y_test, y_pred, sample_weight=test_weights, beta=beta))

    def fit(self, X, y):
        pass

    def predict(self, test_data):
        return test_data[self.feature_list].max(axis=1).tolist()


class bnn_scores:
    feature_model = False

    def __init__(self, name, log_dir):
        self.name = name
        self.log_dir = log_dir
        self.accuracies = []
        self.y_reals = []
        self.predict_probas = []
        self.test_weights = []
        self.precisions = []
        self.recalls = []
        self.fones = []
        self.fbetas = []

    @staticmethod
    def get_y_preds(post_prob_predictions):
        return np.argmax(post_prob_predictions, axis=1)

    def add_cv_scores(self, transformed_train_data, y_train, transformed_test_data, y_test,
                      train_weights=None,
                      test_weights=None):
        y_train_copy = y_train.copy(deep=True)
        y_test_copy = y_test.copy(deep=True)
        import np_bnn as bn

        train_dat = bn.get_data(transformed_train_data, y_train_copy,
                                seed=None,
                                testsize=0,
                                randomize_order=False,
                                instance_id=None, from_file=False)

        bnn_model = bn.npBNN(train_dat,
                             n_nodes=[10, 5],
                             use_class_weights=0,  # set to 1 to use class weights for unbalanced classes
                             actFun=bn.ActFun(fun="tanh"),
                             use_bias_node=-1,
                             # 0) no bias node, 1) bias in input layer, 2) bias in input and hidden layers, 3) bias in input/hidden/output
                             prior_f=1,  # 0) uniform, 1) normal, 2) Cauchy, 3) Laplace
                             p_scale=1,
                             # std for Normal, scale parameter for Cauchy and Laplace, boundaries for Uniform
                             seed=None,
                             init_std=0.1,  # st dev of the initial weights
                             instance_weights=train_weights)

        mcmc = bn.MCMC(bnn_model,
                       n_iteration=100000,  # set to a higher number of more iterations
                       sampling_f=100,  # sampling frequency
                       adapt_f=0.3,  # use adaptive MCMC to target an acceptance rate between 0.3 and 0.6
                       adapt_fM=0.6
                       )

        # initialize output files
        out_file_name = "BNN_outfile_"
        logger = bn.postLogger(bnn_model, wdir=self.log_dir, filename=out_file_name, log_all_weights=0)

        # run MCMC
        bn.run_mcmc(bnn_model, mcmc, logger)

        test_dat = bn.get_data(transformed_test_data, y_test_copy,
                               seed=None,
                               testsize=1,
                               randomize_order=False,
                               instance_id=None, from_file=False)

        # make predictions based on MCMC's estimated weights
        # test data
        post_pr_test = bn.predictBNN(test_dat['test_data'],
                                     pickle_file=logger._pklfile,
                                     test_labels=test_dat['test_labels'],
                                     instance_id=test_dat['id_test_data'],
                                     fname=test_dat['file_name'],
                                     post_summary_mode=0)
        post_prob_predictions = post_pr_test['post_prob_predictions']

        y_pred = self.get_y_preds(post_prob_predictions)
        y_proba = post_prob_predictions[:, 1]
        self.y_reals.append(y_test_copy)
        self.predict_probas.append(y_proba)

        if test_weights is None:
            self.accuracies.append(accuracy_score(y_test_copy, y_pred))
            self.precisions.append(
                precision_score(y_test_copy, y_pred))
            self.fones.append(f1_score(y_test_copy, y_pred))
            self.recalls.append(recall_score(y_test_copy, y_pred))
            self.fbetas.append(fbeta_score(y_test_copy, y_pred, beta=beta))

        else:
            self.accuracies.append(accuracy_score(y_test_copy, y_pred, sample_weight=test_weights))
            self.precisions.append(
                precision_score(y_test_copy, y_pred, sample_weight=test_weights))
            self.fones.append(f1_score(y_test_copy, y_pred, sample_weight=test_weights))
            self.recalls.append(recall_score(y_test_copy, y_pred, sample_weight=test_weights))
            self.fbetas.append(fbeta_score(y_test_copy, y_pred, sample_weight=test_weights, beta=beta))
            self.test_weights.append(test_weights)


class clf_scores:
    feature_model = False
    self_training = False

    def __init__(self, name, clf_class, grid_search_param_grid=None, init_kwargs=None):
        self.name = name
        self.clf_class = clf_class
        self.grid_search_param_grid = grid_search_param_grid
        self.accuracies = []
        self.y_reals = []
        self.predict_probas = []
        self.test_weights = []
        self.precisions = []
        self.recalls = []
        self.fones = []
        self.fbetas = []
        self.init_kwargs = init_kwargs

    def do_grid_search(self, model, X_train, y_train, cv=10, all_train_weight=None):

        gs = GridSearchCV(
            estimator=model,
            param_grid=self.grid_search_param_grid,
            cv=cv,
            n_jobs=-1,
            scoring=make_scorer(get_fbeta_score),
            verbose=1,
            error_score='raise',
            refit=True

        )

        if all_train_weight is not None:

            fitted_model = gs.fit(X_train, y_train, sample_weight=all_train_weight)
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
            self.recalls.append(recall_score(y_test, y_pred))
            self.fbetas.append(fbeta_score(y_test, y_pred, beta=beta))

        else:
            self.accuracies.append(accuracy_score(y_test, y_pred, sample_weight=test_weights))
            self.precisions.append(
                precision_score(y_test, y_pred, sample_weight=test_weights))
            self.fones.append(f1_score(y_test, y_pred, sample_weight=test_weights))
            self.recalls.append(recall_score(y_test, y_pred, sample_weight=test_weights))
            self.fbetas.append(fbeta_score(y_test, y_pred, sample_weight=test_weights, beta=beta))
            self.test_weights.append(test_weights)

        # try:
        #     if self.grid_search_param_grid is not None:
        #         self.temp_feature_importance = dict(zip(clf_instance.best_estimator_.feature_names_in_,
        #                                                 clf_instance.best_estimator_.feature_importances_))
        #     else:
        #         self.temp_feature_importance = dict(
        #             zip(clf_instance.feature_names_in_, clf_instance.feature_importances_))
        # except AttributeError:
        #     try:
        #         # Using coef_ like this assumes model fit on data with standardised parameters.
        #         if self.grid_search_param_grid is not None:
        #             self.temp_feature_importance = dict(zip(clf_instance.best_estimator_.feature_names_in_,
        #                                                     clf_instance.best_estimator_.coef_[0]))
        #         else:
        #             self.temp_feature_importance = dict(zip(clf_instance.feature_names_in_, clf_instance.coef_[0]))
        #     except AttributeError as e:
        #         print(e)
        #         self.temp_feature_importance = dict(zip(clf_instance.feature_names_in_, clf_instance.feature_names_in_))
        # self.feature_importance = {k: [v] for k, v in self.temp_feature_importance.items()}

    def predict_on_unlabelled_data(self, X_train, y_train, unlab_data, train_weights=None):

        if self.init_kwargs is not None:
            clf_instance = self.clf_class(**self.init_kwargs)
        else:
            clf_instance = self.clf_class()

        if self.grid_search_param_grid is not None:
            clf_instance = self.do_grid_search(clf_instance, X_train, y_train, all_train_weight=train_weights)
        else:
            if train_weights is not None:
                clf_instance = clf_instance.fit(X_train, y_train, sample_weight=train_weights)
            else:
                clf_instance = clf_instance.fit(X_train, y_train)

        y_pred = clf_instance.predict(unlab_data)
        y_proba = clf_instance.predict_proba(unlab_data)

        return y_pred, y_proba


def output_scores(models: List[clf_scores], output_dir: str, filetag: str):
    acc_dict = {}
    prec_dict = {}
    recall_dict = {}
    fone_dict = {}
    fbeta_dict = {}

    for model in models:
        acc_dict[model.name] = model.accuracies

        prec_dict[model.name] = model.precisions
        recall_dict[model.name] = model.recalls
        fone_dict[model.name] = model.fones
        fbeta_dict[model.name] = model.fbetas

    acc_df = pd.DataFrame(acc_dict)
    acc_df.describe().to_csv(os.path.join(output_dir, filetag + 'acc_means.csv'))
    acc_df.to_csv(os.path.join(output_dir, filetag + 'acc.csv'))
    output_boxplot(acc_df, os.path.join(output_dir, filetag + 'accuracy_boxplot.jpg'), y_title='Accuracy')

    prec_df = pd.DataFrame(prec_dict)
    prec_df.describe().to_csv(os.path.join(output_dir, filetag + 'prec_means.csv'))
    prec_df.to_csv(os.path.join(output_dir, filetag + 'prec.csv'))
    output_boxplot(prec_df, os.path.join(output_dir, filetag + 'precision_boxplot.jpg'),
                   y_title='Precision')

    recall_df = pd.DataFrame(recall_dict)
    recall_df.describe().to_csv(os.path.join(output_dir, filetag + 'recall_means.csv'))
    recall_df.to_csv(os.path.join(output_dir, filetag + 'recall.csv'))
    output_boxplot(recall_df, os.path.join(output_dir, filetag + 'recall_boxplot.jpg'),
                   y_title='Recall')

    f1_df = pd.DataFrame(fone_dict)
    f1_df.describe().to_csv(os.path.join(output_dir, filetag + 'f1_means.csv'))
    f1_df.to_csv(os.path.join(output_dir, filetag + 'f1.csv'))
    output_boxplot(f1_df, os.path.join(output_dir, filetag + 'f1_boxplot.jpg'),
                   y_title='F1 Score')

    fbeta_df = pd.DataFrame(fbeta_dict)
    fbeta_df.describe().to_csv(os.path.join(output_dir, filetag + 'fb_means.csv'))
    fbeta_df.to_csv(os.path.join(output_dir, filetag + 'fb.csv'))
    output_boxplot(fbeta_df, os.path.join(output_dir, filetag + 'fb_boxplot.jpg'),
                   y_title='F0.5 Score')
    plt.rc('font', size=13)
    # Plot ROC Curves
    for model in models:
        if not model.feature_model:
            all_y_real = np.concatenate(model.y_reals)
            all_y_proba = np.concatenate(model.predict_probas)

            if len(model.test_weights) > 0:
                all_test_weights = np.concatenate(model.test_weights)
                precision, recall, thresholds = precision_recall_curve(all_y_real, all_y_proba,
                                                                       sample_weight=all_test_weights)
                all_ap_score = average_precision_score(all_y_real, all_y_proba,
                                                       sample_weight=all_test_weights)
            else:
                precision, recall, thresholds = precision_recall_curve(all_y_real, all_y_proba)
                all_ap_score = average_precision_score(all_y_real, all_y_proba)
            plt.plot(recall, precision,
                     label=model.name,  # + r': (AP = %0.2f)' % all_ap_score,
                     lw=2, alpha=.8)
            curve_df = pd.DataFrame(
                {'precision': list(precision), 'recall': list(recall), 'thresholds': list(thresholds) + [1]})
            curve_df.to_csv(os.path.join(output_dir, '_'.join([model.name, 'pr_curves.csv'])))

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, '_'.join([filetag, 'pr_curves.jpg'])), dpi=400)
    plt.close()
    plt.cla()
    plt.clf()

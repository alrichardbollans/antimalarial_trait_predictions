import os

import pandas as pd
from pkg_resources import resource_filename
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier

from bias_correction_and_summaries import LABELLED_TRAITS, UNLABELLED_TRAITS, vars_without_target_to_use, \
    known_biasing_features, logit_correction, vars_to_use_in_bias_analysis, \
    ALL_TRAITS, to_target_encode
from general_preprocessing_and_testing import basic_data_prep, clf_scores, FeatureModel, \
    do_basic_preprocessing, output_scores, get_fbeta_score

_output_path = resource_filename(__name__, 'outputs')

_comp_output_dir = os.path.join(_output_path, 'modelling')

prediction_vars_to_use = vars_without_target_to_use
prediction_vars_to_use.remove('In_Malarial_Region')
prediction_vars_to_use.remove('Tested_for_Alkaloids')


def biased_case():
    _bias_model_dir = os.path.join(_comp_output_dir, 'biased_models')
    ### Data
    labelled_data = LABELLED_TRAITS.copy(deep=True)
    labelled_data.reset_index(inplace=True, drop=True)
    X, y = basic_data_prep(labelled_data, prediction_vars_to_use, dropna_rows=False)
    # Check indices are the same
    pd.testing.assert_index_equal(X.index, labelled_data.index)

    unlabelled_data = UNLABELLED_TRAITS.copy(deep=True)
    unlab_X, unlab_y = basic_data_prep(unlabelled_data, prediction_vars_to_use)

    ### Models
    xgb_scores = clf_scores('XGB', XGBClassifier, grid_search_param_grid={'max_depth': [3, 6, 9]
                                                                          },
                            init_kwargs={'eval_metric': get_fbeta_score, 'use_label_encoder': False,
                                         'objective': 'binary:logistic'})
    svc_scores = clf_scores('SVC', SVC,
                            grid_search_param_grid={'C': [0.1, 1, 10],
                                                    'class_weight': ['balanced', None, {0: 0.4, 1: 0.6}]},
                            init_kwargs={'probability': True})
    logit_scores = clf_scores('Logit', LogisticRegression,
                              grid_search_param_grid={'C': [0.1, 1, 10],
                                                      'class_weight': ['balanced', None, {0: 0.4, 1: 0.6}]},
                              init_kwargs={'max_iter': 1000})
    ethnobotanical_scores = FeatureModel('Ethno (M)', ['Antimalarial_Use'])
    general_ethnobotanical_scores = FeatureModel('Ethno (G)', ['Medicinal'])

    models = [xgb_scores, logit_scores, svc_scores, ethnobotanical_scores,
              general_ethnobotanical_scores]

    for i in range(10):

        kf = StratifiedKFold(n_splits=10, shuffle=True)
        for train_index, test_index in kf.split(X, y):
            print(f'{i}th run')
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            imputed_X_train, imputed_X_test, imputed_unlabelled = \
                do_basic_preprocessing(X, y,
                                       train_index,
                                       test_index,
                                       unlab_X,
                                       categorical_features=[
                                           'Family',
                                           'Genus',
                                           'kg_mode'],
                                       impute=True,
                                       scale=True)

            for model in models:
                if model.feature_model:
                    # Use unscaled/unimputed data for these models
                    model.add_cv_scores(X.iloc[test_index], y_test)
                else:
                    model.add_cv_scores(imputed_X_train, y_train, imputed_X_test, y_test)

            output_scores(models, _bias_model_dir, 'biased_case_')


def in_the_wild_test():
    _itw_dir = os.path.join(_comp_output_dir, 'in_the_wild')
    ### Data
    labelled_data = LABELLED_TRAITS.copy(deep=True)
    labelled_data.reset_index(inplace=True, drop=True)
    X, y = basic_data_prep(labelled_data, traits_to_use=prediction_vars_to_use, dropna_rows=False)
    # Check indices are the same
    pd.testing.assert_index_equal(X.index, labelled_data.index)

    unlabelled_data = UNLABELLED_TRAITS.copy(deep=True)
    unlab_X, unlab_y = basic_data_prep(unlabelled_data, traits_to_use=prediction_vars_to_use)

    ### Models

    xgb_scores = clf_scores('XGB', XGBClassifier, grid_search_param_grid={
        'max_depth': [3, 6, 9]
    }, init_kwargs={'eval_metric': get_fbeta_score, 'use_label_encoder': False, 'objective': 'binary:logistic'})
    svc_scores = clf_scores('SVC', SVC, grid_search_param_grid={'C': [0.1, 1, 10],
                                                                'class_weight': ['balanced', None, {0: 0.4, 1: 0.6}]},
                            init_kwargs={'probability': True})
    logit_scores = clf_scores('Logit', LogisticRegression,
                              grid_search_param_grid={'C': [0.1, 1, 10],
                                                      'class_weight': ['balanced', None, {0: 0.4, 1: 0.6}]},
                              init_kwargs={'max_iter': 1000})

    ethnobotanical_scores = FeatureModel('Ethno (M)', ['Antimalarial_Use'])
    general_ethnobotanical_scores = FeatureModel('Ethno (G)', ['Medicinal'])

    models = [xgb_scores, logit_scores, svc_scores,
              ethnobotanical_scores,
              general_ethnobotanical_scores]
    all_weights = \
        logit_correction(labelled_data, ALL_TRAITS,
                         selection_vars=known_biasing_features,
                         cols_to_target_encode=to_target_encode)[
            'weight']
    for i in range(10):
        kf = StratifiedKFold(n_splits=10, shuffle=True)
        for train_index, test_index in kf.split(X, y):
            print(f'{i}th run')
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            imputed_X_train, imputed_X_test, imputed_unlabelled = \
                do_basic_preprocessing(X, y,
                                       train_index,
                                       test_index,
                                       unlab_X,
                                       categorical_features=[
                                           'Family',
                                           'Genus',
                                           'kg_mode'],
                                       impute=True,
                                       scale=True)

            train_weights = all_weights.iloc[train_index]
            test_weights = all_weights.iloc[test_index]

            for model in models:
                if model.feature_model:
                    # Use unscaled/unimputed data for these models
                    model.add_cv_scores(X.iloc[test_index], y_test,
                                        test_weights=test_weights)
                else:
                    model.add_cv_scores(imputed_X_train, y_train, imputed_X_test, y_test, train_weights=train_weights,
                                        test_weights=test_weights)
            output_scores(models, _itw_dir, 'logit_')


def check_twinning():
    labelled_data = LABELLED_TRAITS.copy(deep=True)
    labelled_data.reset_index(inplace=True, drop=True)
    X, y = basic_data_prep(labelled_data, prediction_vars_to_use, dropna_rows=False)
    dup_df1 = X[X.duplicated()]
    if len(dup_df1.index) > 0:
        raise ValueError(f'Twins present')


def main():
    check_twinning()
    biased_case()
    in_the_wild_test()
    #

    # Write variables used
    with open(os.path.join(_comp_output_dir, 'variable_docs.txt'), 'w') as the_file:
        the_file.write(f'vars_to_use_in_bias_analysis:{vars_to_use_in_bias_analysis}\n')
        the_file.write(f'prediction_vars_to_use:{prediction_vars_to_use}\n')
        the_file.write(f'analysis_vars_not_in_predictions:{[c for c in vars_to_use_in_bias_analysis if c not in prediction_vars_to_use]}\n')
        the_file.write(f'pred_vars_not_in_analysis:{[c for c in prediction_vars_to_use if c not in vars_to_use_in_bias_analysis]}\n')


if __name__ == '__main__':
    main()

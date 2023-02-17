import os

import numpy as np
import pandas as pd
import scipy
from pkg_resources import resource_filename
from sklearn.svm import SVC

from antimalarial_predictions import prediction_vars_to_use
from bias_correction_and_summaries import LABELLED_TRAITS, UNLABELLED_TRAITS, all_features_to_target_encode, \
    WEIGHTED_LABELLED_DATA, WEIGHTED_UNLABELLED_DATA
from general_preprocessing_and_testing import basic_data_prep, clf_scores, do_basic_preprocessing
from import_trait_data import TARGET_COLUMN

_output_path = resource_filename(__name__, 'outputs')

_predictions_output_dir = os.path.join(_output_path, 'predictions')
_unlabelled_output_csv = os.path.join(_predictions_output_dir, 'unlabelled_output.csv')


def make_predictions():
    assert 'In_Malarial_Region' not in prediction_vars_to_use
    assert 'Tested_for_Alkaloids' not in prediction_vars_to_use
    ### Data
    labelled_data = LABELLED_TRAITS.copy(deep=True)
    weighted_labelled_df = pd.read_csv(WEIGHTED_LABELLED_DATA, index_col=0)
    all_weights = weighted_labelled_df['weight']
    X, y = basic_data_prep(labelled_data, prediction_vars_to_use, dropna_rows=False)
    # Check indices are the same
    pd.testing.assert_index_equal(X.index, labelled_data.index)
    pd.testing.assert_index_equal(all_weights.index, labelled_data.index)

    unlabelled_data = UNLABELLED_TRAITS.copy(deep=True)
    unlab_X, unlab_y = basic_data_prep(unlabelled_data, prediction_vars_to_use)

    # Just use best performing model
    svc_scores = clf_scores(chosen_model, SVC, grid_search_param_grid={'C': [0.1, 1, 10],
                                                                       'class_weight': ['balanced', None,
                                                                                        {0: 0.4, 1: 0.6}]},
                            init_kwargs={'probability': True})

    models = [svc_scores]

    imputed_X_train, imputed_X_test, imputed_unlabelled = \
        do_basic_preprocessing(X, y,
                               unlabelled_data=unlab_X,
                               categorical_features=all_features_to_target_encode,
                               impute=True,
                               scale=True,
                               PCA_cont_vars=True)
    pd.testing.assert_index_equal(unlabelled_data.index, imputed_unlabelled.index)
    for model in models:
        y_pred, y_proba = model.predict_on_unlabelled_data(imputed_X_train, y,
                                                           imputed_unlabelled,
                                                           train_weights=all_weights)

        unlabelled_data[model.name + ' Probability Estimate'] = y_proba[:, 1]
        unlabelled_data[model.name + ' Prediction'] = y_pred

    unlabelled_data.to_csv(_unlabelled_output_csv)


def compare_to_selection_probability():
    weighted_unlabelled_df = pd.read_csv(WEIGHTED_UNLABELLED_DATA)
    unlabelled_predictions = pd.read_csv(_unlabelled_output_csv)

    unlabelled_predictions = unlabelled_predictions[
        ['Accepted_Name', chosen_model + ' Probability Estimate', chosen_model + ' Prediction']]

    df_for_analysis = pd.merge(unlabelled_predictions, weighted_unlabelled_df, on='Accepted_Name')

    # surprises
    weighted_labelled_df = pd.read_csv(WEIGHTED_LABELLED_DATA)

    fifty_quantile = weighted_unlabelled_df.quantile()
    value = fifty_quantile['P(s|x)']
    surprises = df_for_analysis[df_for_analysis['P(s|x)'] < value]
    surprises_rate = surprises[chosen_model + ' Prediction'].mean()
    number_of_surprises = surprises[chosen_model + ' Prediction'].sum()
    len_suprises_check = len(surprises.index)

    labelled_surprises = weighted_labelled_df[weighted_labelled_df['P(s|x)'] < value]
    labelled_surprises_rate = labelled_surprises[TARGET_COLUMN].mean()
    labelled_number_of_surprises = labelled_surprises[TARGET_COLUMN].sum()
    labelled_len_suprises_check = len(labelled_surprises.index)

    data = pd.read_csv(os.path.join(_output_path, 'in_the_wild', '10_10', 'itw_prec.csv'))[chosen_model]
    data = (data,)
    bootstrap_result = scipy.stats.bootstrap(data, np.mean, confidence_level=0.95)

    score_interval = bootstrap_result.confidence_interval
    lower_lim = score_interval.low
    upper_lim = score_interval.high
    print((lower_lim, upper_lim))

    pd.DataFrame(
        {'num': [number_of_surprises, number_of_surprises * upper_lim, number_of_surprises * lower_lim,
                 labelled_number_of_surprises],
         'rate': [surprises_rate, surprises_rate * upper_lim, surprises_rate * lower_lim,
                  labelled_surprises_rate],
         'len': [len_suprises_check, len_suprises_check, len_suprises_check, labelled_len_suprises_check]},
        index=['unlabelled', 'upperlim', 'lowerlim', 'labelled']).to_csv(
        os.path.join(_predictions_output_dir, 'surprises.csv'))


def main():
    make_predictions()
    compare_to_selection_probability()


if __name__ == '__main__':
    chosen_model = 'SVC'
    main()

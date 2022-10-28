import os

import pandas as pd
from pkg_resources import resource_filename
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, recall_score
from sklearn.model_selection import StratifiedKFold

from fixing_bias import known_biasing_features, LABELLED_TRAITS_IN_ALL_REGIONS, \
    UNLABELLED_TRAITS_IN_ALL_REGIONS, bias_output_dir, vars_without_target_to_use
from general_preprocessing_and_testing import output_boxplot, do_basic_preprocessing

_inputs_path = resource_filename(__name__, 'inputs')

_temp_outputs_path = resource_filename(__name__, 'temp_outputs')

_output_path = resource_filename(__name__, 'outputs')
if not os.path.isdir(_inputs_path):
    os.mkdir(_inputs_path)
if not os.path.isdir(_temp_outputs_path):
    os.mkdir(_temp_outputs_path)
if not os.path.isdir(_output_path):
    os.mkdir(_output_path)


def logit_test():
    labelled_data = LABELLED_TRAITS_IN_ALL_REGIONS.copy(deep=True)
    unlabelled_data = UNLABELLED_TRAITS_IN_ALL_REGIONS.copy(deep=True)
    labelled_data['selected'] = 1
    unlabelled_data['selected'] = 0

    all_data = pd.concat([labelled_data, unlabelled_data])

    X = all_data[vars_without_target_to_use]
    y = all_data[['selected']]

    logit_accs = []
    logit_briers = []
    logit_recalls= []

    # if any(x in TRAITS_WITH_NANS for x in known_biasing_features):
    #     # In the following we won't impute, if there are traits with nans we need to change this
    #     raise ValueError

    for i in range(2):
        kf = StratifiedKFold(n_splits=10, shuffle=True)
        for train_index, test_index in kf.split(X, y):
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            processed_X_train, processed_X_test, processed_unlabelled, cols_with_good_variance = \
                do_basic_preprocessing(X, y,
                                       train_index,
                                       test_index,
                                       unlabelled_data=None,
                                       categorical_features=[
                                           'Family',
                                           'Genus',
                                           'kg_mode'],
                                       impute=False,
                                       scale=True)

            logit = LogisticRegression()

            logit.fit(processed_X_train[known_biasing_features], y_train)

            y_pred = logit.predict(processed_X_test[known_biasing_features])
            logit_accs.append(accuracy_score(y_test, y_pred))
            logit_recalls.append(recall_score(y_test, y_pred))

            logit_prob_estimates = logit.predict_proba(processed_X_test[known_biasing_features])[:, 1]

            logit_briers.append(brier_score_loss(y_test, logit_prob_estimates, pos_label=1))



            import os
            acc_dict = {}

            acc_dict['logit'] = logit_accs

            acc_df = pd.DataFrame(acc_dict)
            acc_df.describe().to_csv(os.path.join(bias_output_dir, 'logit_test', 'acc_means.csv'))
            acc_df.to_csv(os.path.join(bias_output_dir, 'logit_test', 'acc.csv'))
            output_boxplot(acc_df, os.path.join(bias_output_dir, 'logit_test', 'accuracy_boxplot.png'),
                           y_title='Model Accuracy')

            recall_dict = {}

            recall_dict['logit'] = logit_recalls

            recall_df = pd.DataFrame(recall_dict)
            recall_df.describe().to_csv(os.path.join(bias_output_dir, 'logit_test', 'recall_means.csv'))
            recall_df.to_csv(os.path.join(bias_output_dir, 'logit_test', 'recall.csv'))
            output_boxplot(recall_df, os.path.join(bias_output_dir, 'logit_test', 'recall_boxplot.png'),
                           y_title='Model Recall')

            brie_dict = {}
            brie_dict['logit'] = logit_briers

            brie_df = pd.DataFrame(brie_dict)
            brie_df.describe().to_csv(os.path.join(bias_output_dir, 'logit_test', 'br_means.csv'))
            brie_df.to_csv(os.path.join(bias_output_dir, 'logit_test', 'br.csv'))
            output_boxplot(brie_df, os.path.join(bias_output_dir, 'logit_test', 'br_boxplot.png'),
                           y_title='Model Brier Score')


if __name__ == '__main__':
    logit_test()

import os
from typing import List

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, recall_score
from sklearn.model_selection import StratifiedKFold

from bias_correction_and_summaries import apriori_known_biasing_features, LABELLED_TRAITS, \
    UNLABELLED_TRAITS, bias_output_dir, vars_without_target_to_use, all_features_to_target_encode
from general_preprocessing_and_testing import output_boxplot, do_basic_preprocessing


def logit_test_instance(features_to_use: List[str], processed_X_train, y_train, processed_X_test, y_test):
    logit = LogisticRegression()
    logit.fit(processed_X_train[features_to_use], y_train)
    y_pred = logit.predict(processed_X_test[features_to_use])
    logit_acc = accuracy_score(y_test, y_pred)
    logit_recall = recall_score(y_test, y_pred)

    logit_prob_estimates = logit.predict_proba(processed_X_test[features_to_use])[:, 1]

    logit_brier = brier_score_loss(y_test, logit_prob_estimates, pos_label=1)

    return logit_acc, logit_recall, logit_brier


def main():
    labelled_data = LABELLED_TRAITS.copy(deep=True)
    unlabelled_data = UNLABELLED_TRAITS.copy(deep=True)
    labelled_data['selected'] = 1
    unlabelled_data['selected'] = 0

    all_data = pd.concat([labelled_data, unlabelled_data])

    X = all_data[vars_without_target_to_use]
    y = all_data[['selected']]

    apriori_accs, apriori_recalls, apriori_briers = [], [], []
    all_accs, all_recalls, all_briers = [], [], []

    for i in range(10):
        kf = StratifiedKFold(n_splits=10, shuffle=True)
        for train_index, test_index in kf.split(X, y):
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # From Elements of Statistical Learning, recommended to scale inputs
            processed_X_train, processed_X_test, processed_unlabelled = \
                do_basic_preprocessing(X, y,
                                       train_index,
                                       test_index,
                                       unlabelled_data=None,
                                       categorical_features=all_features_to_target_encode,
                                       impute=True,
                                       scale=True)
            ## Restricted to a priori biased features
            apriori_acc, apriori_recall, apriori_brier = logit_test_instance(
                apriori_known_biasing_features, processed_X_train, y_train, processed_X_test,
                y_test)
            apriori_accs.append(apriori_acc)
            apriori_recalls.append(apriori_recall)
            apriori_briers.append(apriori_brier)

            ## All features
            all_acc, all_recall, all_brier = logit_test_instance(
                vars_without_target_to_use, processed_X_train, y_train, processed_X_test,
                y_test)
            all_accs.append(all_acc)
            all_recalls.append(all_recall)
            all_briers.append(all_brier)

            ## Output
            acc_dict = {}
            acc_dict['apriori_features'] = apriori_accs
            acc_dict['all_features'] = all_accs

            acc_df = pd.DataFrame(acc_dict)
            acc_df.describe().to_csv(os.path.join(bias_output_dir, 'logit_test', 'acc_means.csv'))
            acc_df.to_csv(os.path.join(bias_output_dir, 'logit_test', 'acc.csv'))
            output_boxplot(acc_df, os.path.join(bias_output_dir, 'logit_test', 'accuracy_boxplot.png'),
                           y_title='Model Accuracy')

            recall_dict = {}
            recall_dict['apriori_features'] = apriori_recalls
            recall_dict['all_features'] = all_recalls

            recall_df = pd.DataFrame(recall_dict)
            recall_df.describe().to_csv(os.path.join(bias_output_dir, 'logit_test', 'recall_means.csv'))
            recall_df.to_csv(os.path.join(bias_output_dir, 'logit_test', 'recall.csv'))
            output_boxplot(recall_df, os.path.join(bias_output_dir, 'logit_test', 'recall_boxplot.png'),
                           y_title='Model Recall')

            brie_dict = {}
            brie_dict['apriori_features'] = apriori_briers
            brie_dict['all_features'] = all_briers

            brie_df = pd.DataFrame(brie_dict)
            brie_df.describe().to_csv(os.path.join(bias_output_dir, 'logit_test', 'br_means.csv'))
            brie_df.to_csv(os.path.join(bias_output_dir, 'logit_test', 'br.csv'))
            output_boxplot(brie_df, os.path.join(bias_output_dir, 'logit_test', 'br_boxplot.png'),
                           y_title='Model Brier Score')


if __name__ == '__main__':
    main()

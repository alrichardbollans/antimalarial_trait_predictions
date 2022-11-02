import os

import pandas as pd

from bias_correction_and_summaries import oversample_by_weight, LABELLED_TRAITS, bias_output_dir, \
    known_biasing_features, UNLABELLED_TRAITS, to_target_encode
from import_trait_data import TARGET_COLUMN

metric_output_dir = os.path.join(bias_output_dir, 'screening_comparison')


def estimate_approach_accuracy(df: pd.DataFrame, feature1, feature2=None):
    if feature2 is None:
        true_positives = len(df[(df[feature1] == 1) & (df[TARGET_COLUMN] == 1)].index)
        true_negatives = len(df[(df[feature1] == 0) & (df[TARGET_COLUMN] == 0)].index)
    else:
        true_positives = len(df[(df[feature1] == 1) & (df[feature2] == 1) & (df[TARGET_COLUMN] == 1)].index)
        true_negatives = len(df[(df[feature1] == 0) & (df[feature2] == 0) & (df[TARGET_COLUMN] == 0)].index)

    accuracy = float(true_positives + true_negatives) / len(df.index)
    print(accuracy)
    return accuracy


def estimate_population_activity(df: pd.DataFrame):
    num_active_samples = len(df[df[TARGET_COLUMN] == 1].index)
    active_probability = float(num_active_samples) / len(df.index)
    print(active_probability)
    return active_probability


def get_accuracies(corrected_df: pd.DataFrame):
    accuracies = []

    print('Random Approach:')
    accuracies.append(0.5)

    print('Ethno Approach Acc:')
    accuracies.append(estimate_approach_accuracy(corrected_df, 'Medicinal'))

    print('Ethno(AntiMal) Approach Acc:')
    accuracies.append(estimate_approach_accuracy(corrected_df, 'Antimalarial_Use'))

    return accuracies


def get_precisions(corrected_df: pd.DataFrame):
    precisions = []

    print('Random Approach:')
    precisions.append(estimate_population_activity(corrected_df))

    print('Ethno Approach Precision:')
    ethno_df = corrected_df[corrected_df['Medicinal'] == 1]
    precisions.append(estimate_population_activity(ethno_df))

    print('Ethno(AntiMal) Approach Precision:')
    antimal_df = corrected_df[corrected_df['Antimalarial_Use'] == 1]
    precisions.append(estimate_population_activity(antimal_df))

    return precisions


def get_model_precisions():
    approaches = ['Random', 'Ethno (G)', 'Ethno (M)']

    logit_corrected_df = oversample_by_weight(LABELLED_TRAITS, UNLABELLED_TRAITS, 'logit',
                                              known_biasing_features, cols_to_target_encode=to_target_encode
                                              )
    logit_corrected_precisions = get_precisions(logit_corrected_df)

    unadjusted_precisions = get_precisions(LABELLED_TRAITS)

    out = pd.DataFrame(
        {'Uncorrected': unadjusted_precisions,  # 'Ratio Corrected': ratio_corrected_precisions,
         'Logit Corrected': logit_corrected_precisions},
        index=approaches)
    out.to_csv(os.path.join(metric_output_dir, 'precisions.csv'))


def get_model_accuracies():
    approaches = ['Random', 'Ethno (G)', 'Ethno (M)']

    logit_corrected_df = oversample_by_weight(LABELLED_TRAITS, UNLABELLED_TRAITS, 'logit',
                                              known_biasing_features, cols_to_target_encode=to_target_encode
                                              )
    logit_corrected_accs = get_accuracies(logit_corrected_df)

    unadjusted_accs = get_accuracies(LABELLED_TRAITS)

    out = pd.DataFrame(
        {'Uncorrected': unadjusted_accs,
         'Logit Corrected': logit_corrected_accs},
        index=approaches)
    out.to_csv(os.path.join(metric_output_dir, 'accuracies.csv'))


def main():
    # Estimate model performance in all regions
    get_model_precisions()
    get_model_accuracies()


if __name__ == '__main__':
    main()

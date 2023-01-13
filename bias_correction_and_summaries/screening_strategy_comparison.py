import os

import numpy as np
import pandas as pd

from bias_correction_and_summaries import LABELLED_TRAITS, bias_output_dir, \
    WEIGHTED_LABELLED_DATA
from import_trait_data import TARGET_COLUMN

metric_output_dir = os.path.join(bias_output_dir, 'screening_comparison')


def add_approach_result_column(df: pd.DataFrame, feature: str):
    out_df = df.copy(deep=True)
    out_df['result'] = np.where(out_df[feature] == out_df[TARGET_COLUMN], 1, 0)
    out_df = out_df[[feature, TARGET_COLUMN, 'result', 'weight']]
    return out_df


def get_model_precisions():
    weighted_labelled_df = pd.read_csv(WEIGHTED_LABELLED_DATA, index_col=0)
    antimal_df = weighted_labelled_df[weighted_labelled_df['Antimalarial_Use'] == 1]
    antimal_values = [
        antimal_df[TARGET_COLUMN].mean(),
        np.average(antimal_df[TARGET_COLUMN], weights=antimal_df['weight'])
    ]

    medicinal_df = weighted_labelled_df[weighted_labelled_df['Medicinal'] == 1]
    medicinal_values = [
        medicinal_df[TARGET_COLUMN].mean(),
        np.average(medicinal_df[TARGET_COLUMN], weights=medicinal_df['weight'])
    ]

    total_corrected_mean = np.average(weighted_labelled_df[TARGET_COLUMN],
                                      weights=weighted_labelled_df['weight'])
    all_values = [weighted_labelled_df[TARGET_COLUMN].mean(),
                  total_corrected_mean
                  ]
    out = pd.DataFrame(
        {'Random': all_values, 'Ethno (G)': medicinal_values, 'Ethno (M)': antimal_values},
        index=['Uncorrected', 'Logit Corrected'])
    out = out.transpose()
    out.to_csv(os.path.join(metric_output_dir, 'precisions.csv'))


def get_model_accuracies():
    weighted_labelled_df = pd.read_csv(WEIGHTED_LABELLED_DATA, index_col=0)
    antimal_df = add_approach_result_column(weighted_labelled_df, 'Antimalarial_Use')
    antimal_values = [
        antimal_df['result'].mean(),
        np.average(antimal_df['result'], weights=antimal_df['weight'])
    ]

    medicinal_df = add_approach_result_column(weighted_labelled_df, 'Medicinal')
    medicinal_values = [
        medicinal_df['result'].mean(),
        np.average(medicinal_df['result'], weights=medicinal_df['weight'])
    ]

    out = pd.DataFrame(
        {'Random': [0.5, 0.5], 'Ethno (G)': medicinal_values, 'Ethno (M)': antimal_values},
        index=['Uncorrected', 'Logit Corrected'])
    out = out.transpose()
    out.to_csv(os.path.join(metric_output_dir, 'accuracies.csv'))


def main():
    # Estimate model performance in all regions
    get_model_precisions()
    get_model_accuracies()


if __name__ == '__main__':
    main()

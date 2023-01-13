import os

import numpy as np
import pandas as pd

from bias_correction_and_summaries import bias_output_dir, ALL_TRAITS, WEIGHTED_LABELLED_DATA
from import_trait_data import TARGET_COLUMN

summary_output_dir = os.path.join(bias_output_dir, 'corrected_summaries')


def summarise_activites():
    weighted_labelled_df = pd.read_csv(WEIGHTED_LABELLED_DATA, index_col=0)
    apoc_df = weighted_labelled_df[weighted_labelled_df['Family'] == 'Apocynaceae']
    apoc_family_values = [
        apoc_df[TARGET_COLUMN].mean(),
        np.average(apoc_df[TARGET_COLUMN], weights=apoc_df['weight'])
    ]
    rub_df = weighted_labelled_df[weighted_labelled_df['Family'] == 'Rubiaceae']
    rub_family_values = [
        rub_df[
            TARGET_COLUMN].mean(),
        np.average(rub_df[TARGET_COLUMN], weights=rub_df['weight'])
    ]
    log_df = weighted_labelled_df[weighted_labelled_df['Family'] == 'Loganiaceae']
    logan_family_values = [
        log_df[TARGET_COLUMN].mean(),
        np.average(log_df[TARGET_COLUMN], weights=log_df['weight'])
    ]

    total_corrected_mean = np.average(weighted_labelled_df[TARGET_COLUMN],
                                      weights=weighted_labelled_df['weight'])
    all_family_values = [weighted_labelled_df[TARGET_COLUMN].mean(),
                         total_corrected_mean
                         ]
    out = pd.DataFrame(
        {'Apocynaceae': apoc_family_values, 'Loganiaceae': logan_family_values,
         'Rubiaceae': rub_family_values,
         'All': all_family_values},
        index=['Uncorrected', 'Logit Corrected'])
    out = out.transpose()
    out.to_csv(os.path.join(summary_output_dir, 'corrected_activities.csv'))
    num_active_sp = pd.DataFrame({'Estimated Number Active Species': [
        total_corrected_mean * len(ALL_TRAITS.index)]})
    num_active_sp.to_csv(os.path.join(summary_output_dir, 'Estimated Number Active Species.csv'))
    print(len(ALL_TRAITS.index))
    print(total_corrected_mean)
    print(weighted_labelled_df[TARGET_COLUMN].sum())


def main():
    summarise_activites()


if __name__ == '__main__':
    main()

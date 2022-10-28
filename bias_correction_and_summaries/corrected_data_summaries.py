import os

import pandas as pd

from bias_correction_and_summaries import oversample_by_weight, bias_output_dir, LABELLED_TRAITS_IN_ALL_REGIONS, \
    known_biasing_features, UNLABELLED_TRAITS_IN_ALL_REGIONS, to_target_encode
from import_trait_data import TARGET_COLUMN

summary_output_dir = os.path.join(bias_output_dir, 'corrected_summaries')


def summarise_activites():
    logit_corrected_df = oversample_by_weight(LABELLED_TRAITS_IN_ALL_REGIONS, UNLABELLED_TRAITS_IN_ALL_REGIONS, 'logit',
                                              known_biasing_features, cols_to_target_encode=to_target_encode)
    apoc_family_values = [
        LABELLED_TRAITS_IN_ALL_REGIONS[LABELLED_TRAITS_IN_ALL_REGIONS['Family'] == 'Apocynaceae'][
            TARGET_COLUMN].mean(),
        # ratio_corrected_df[ratio_corrected_df['Family'] == 'Apocynaceae'][TARGET_COLUMN].mean(),
        logit_corrected_df[logit_corrected_df['Family'] == 'Apocynaceae'][TARGET_COLUMN].mean(),
    ]
    rub_family_values = [
        LABELLED_TRAITS_IN_ALL_REGIONS[LABELLED_TRAITS_IN_ALL_REGIONS['Family'] == 'Rubiaceae'][
            TARGET_COLUMN].mean(),
        # ratio_corrected_df[ratio_corrected_df['Family'] == 'Rubiaceae'][TARGET_COLUMN].mean(),
        logit_corrected_df[logit_corrected_df['Family'] == 'Rubiaceae'][TARGET_COLUMN].mean(),
    ]
    logan_family_values = [
        LABELLED_TRAITS_IN_ALL_REGIONS[LABELLED_TRAITS_IN_ALL_REGIONS['Family'] == 'Loganiaceae'][
            TARGET_COLUMN].mean(),
        # ratio_corrected_df[ratio_corrected_df['Family'] == 'Loganiaceae'][TARGET_COLUMN].mean(),
        logit_corrected_df[logit_corrected_df['Family'] == 'Loganiaceae'][TARGET_COLUMN].mean(),
    ]

    all_family_values = [LABELLED_TRAITS_IN_ALL_REGIONS[TARGET_COLUMN].mean(),
                         logit_corrected_df[TARGET_COLUMN].mean(),
                         ]
    out = pd.DataFrame(
        {'Apocynaceae': apoc_family_values, 'Loganiaceae': logan_family_values, 'Rubiaceae': rub_family_values,
         'All': all_family_values},
        index=['Uncorrected', 'Logit Corrected'])
    out = out.transpose()
    out.to_csv(os.path.join(summary_output_dir, 'corrected_activities.csv'))



def main():
    summarise_activites()


if __name__ == '__main__':
    main()

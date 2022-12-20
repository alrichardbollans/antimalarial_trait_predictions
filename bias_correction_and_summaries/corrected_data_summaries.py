import os

import pandas as pd

from bias_correction_and_summaries import oversample_by_weight, bias_output_dir, LABELLED_TRAITS, \
    UNLABELLED_TRAITS, ALL_TRAITS
from import_trait_data import TARGET_COLUMN

summary_output_dir = os.path.join(bias_output_dir, 'corrected_summaries')


def summarise_activites():
    logit_corrected_df = oversample_by_weight(LABELLED_TRAITS, UNLABELLED_TRAITS)
    apoc_family_values = [
        LABELLED_TRAITS[LABELLED_TRAITS['Family'] == 'Apocynaceae'][
            TARGET_COLUMN].mean(),
        logit_corrected_df[logit_corrected_df['Family'] == 'Apocynaceae'][TARGET_COLUMN].mean()
    ]
    rub_family_values = [
        LABELLED_TRAITS[LABELLED_TRAITS['Family'] == 'Rubiaceae'][
            TARGET_COLUMN].mean(),
        logit_corrected_df[logit_corrected_df['Family'] == 'Rubiaceae'][TARGET_COLUMN].mean()
    ]
    logan_family_values = [
        LABELLED_TRAITS[LABELLED_TRAITS['Family'] == 'Loganiaceae'][
            TARGET_COLUMN].mean(),
        logit_corrected_df[logit_corrected_df['Family'] == 'Loganiaceae'][TARGET_COLUMN].mean()
    ]

    all_family_values = [LABELLED_TRAITS[TARGET_COLUMN].mean(),
                         logit_corrected_df[TARGET_COLUMN].mean()
                         ]
    out = pd.DataFrame(
        {'Apocynaceae': apoc_family_values, 'Loganiaceae': logan_family_values,
         'Rubiaceae': rub_family_values,
         'All': all_family_values},
        index=['Uncorrected', 'Logit Corrected'])
    out = out.transpose()
    out.to_csv(os.path.join(summary_output_dir, 'corrected_activities.csv'))
    num_active_sp = pd.DataFrame({'Estimated Number Active Species':[logit_corrected_df[TARGET_COLUMN].mean() * len(ALL_TRAITS.index)]})
    num_active_sp.to_csv(os.path.join(summary_output_dir, 'Estimated Number Active Species.csv'))
    print(len(ALL_TRAITS.index))
    print(logit_corrected_df[TARGET_COLUMN].mean())
    print(logit_corrected_df[TARGET_COLUMN].sum())
def main():
    summarise_activites()



if __name__ == '__main__':
    main()

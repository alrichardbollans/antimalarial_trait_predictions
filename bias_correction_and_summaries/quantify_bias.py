import os

import pandas as pd

from bias_correction_and_summaries import bias_output_dir, LABELLED_TRAITS_IN_ALL_REGIONS, \
    ALL_TRAITS_IN_ALL_REGIONS, UNLABELLED_TRAITS_IN_ALL_REGIONS, \
    vars_to_use_in_bias_analysis
from import_trait_data import DISCRETE_VARS, \
    CONTINUOUS_VARS
from stats_methods import compare_sample_distributions, plot_means_all_vs_labelled, holm_bonferroni_correction

quantbias_output_dir = os.path.join(bias_output_dir, 'quantifying bias')


def quantify_given_bias():
    df_all_regions = compare_sample_distributions(LABELLED_TRAITS_IN_ALL_REGIONS, ALL_TRAITS_IN_ALL_REGIONS,
                                                  output_csv=os.path.join(quantbias_output_dir, 'given_bias.csv')
                                                  )
    cont_df = df_all_regions[~df_all_regions['ks_p_value'].isna()]
    cont_df = cont_df[['Feature', 'ks_p_value']].rename(columns={'ks_p_value': 'p_value'})
    disc_df = df_all_regions[df_all_regions['ks_p_value'].isna()]
    disc_df = disc_df[['Feature', 'chi2_p']].rename(columns={'chi2_p': 'p_value'})

    holm_df = pd.concat([cont_df, disc_df])
    holm_df = holm_bonferroni_correction(holm_df, 'p_value')
    holm_df.to_csv(os.path.join(quantbias_output_dir, 'given_bias_corrected.csv'))


def summarise_traits():
    """
    Gets summaries of different datasets
    """
    LABELLED_TRAITS_IN_ALL_REGIONS.describe().to_csv(os.path.join(quantbias_output_dir, 'labelled trait summary.csv'))
    UNLABELLED_TRAITS_IN_ALL_REGIONS.describe().to_csv(
        os.path.join(quantbias_output_dir, 'unlabelled trait summary.csv'))
    ALL_TRAITS_IN_ALL_REGIONS.describe().to_csv(os.path.join(quantbias_output_dir, 'all trait summary.csv'))

    # b_df = oversample_by_weight(LABELLED_TRAITS_IN_ALL_REGIONS, allow_data_leak=True, method='ratio')
    # b_df.describe().to_csv(os.path.join(quantbias_output_dir, 'corrected trait summary.csv'))


def plot_data_means():
    discrete_traits = [c for c in vars_to_use_in_bias_analysis if c in DISCRETE_VARS]
    plot_means_all_vs_labelled(ALL_TRAITS_IN_ALL_REGIONS, discrete_traits,
                               os.path.join(quantbias_output_dir, 'discrete_means.png'), minmaxscale=True)

    cont_traits = [c for c in vars_to_use_in_bias_analysis if c in CONTINUOUS_VARS]
    plot_means_all_vs_labelled(ALL_TRAITS_IN_ALL_REGIONS, cont_traits,
                               os.path.join(quantbias_output_dir, 'cont_means.png'), minmaxscale=True)

    print('non numeric traits in bias analysis:')
    print([x for x in vars_to_use_in_bias_analysis if x not in discrete_traits and x not in cont_traits])


def main():
    quantify_given_bias()
    summarise_traits()
    plot_data_means()

    # Write variables used
    with open(os.path.join(bias_output_dir, 'variable_docs.txt'), 'w') as the_file:
        the_file.write(f'vars_to_use_in_bias_analysis:{vars_to_use_in_bias_analysis}\n')


if __name__ == '__main__':
    main()

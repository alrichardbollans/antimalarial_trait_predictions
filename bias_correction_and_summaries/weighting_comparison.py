import os
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pkg_resources import resource_filename

from fixing_bias import oversample_by_weight, ALL_TRAITS_IN_ALL_REGIONS, LABELLED_TRAITS_IN_ALL_REGIONS, \
    vars_to_use_in_bias_analysis, known_biasing_features, UNLABELLED_TRAITS_IN_ALL_REGIONS, to_target_encode
from import_trait_data import BINARY_VARS, NUMERIC_TRAITS, CONTINUOUS_VARS, HABIT_COLS, DISCRETE_VARS

_output_path = resource_filename(__name__, 'outputs')

_comparison_output_dir = os.path.join(_output_path, 'weighting_comparison')

if not os.path.isdir(_output_path):
    os.mkdir(_output_path)


def plot_corrected_means(vars_to_compare: List[str], out_filename: str):
    from sklearn.preprocessing import MinMaxScaler

    logit_corrected_df = \
        oversample_by_weight(LABELLED_TRAITS_IN_ALL_REGIONS, UNLABELLED_TRAITS_IN_ALL_REGIONS, 'logit',
                             known_biasing_features, to_target_encode)[
            vars_to_compare]

    all_traits = ALL_TRAITS_IN_ALL_REGIONS[vars_to_compare]
    labelled_traits = LABELLED_TRAITS_IN_ALL_REGIONS[vars_to_compare]
    # scale_features for plotting
    s = MinMaxScaler()
    s.fit(all_traits)
    all_traits = pd.DataFrame(s.transform(all_traits),
                              columns=s.get_feature_names_out())
    labelled_traits = pd.DataFrame(s.transform(labelled_traits),
                                   columns=s.get_feature_names_out())

    logit_corrected_df = pd.DataFrame(s.transform(logit_corrected_df),
                                      columns=s.get_feature_names_out())

    labelled_means = labelled_traits.describe().loc[['mean']].values.tolist()[0]
    all_means = all_traits.describe().loc[['mean']].values.tolist()[0]

    logit_corrected_means = logit_corrected_df.describe().loc[['mean']].values.tolist()[0]


    plt.rcParams['figure.figsize'] = [10, 6]
    width = 0.2
    X_axis = np.arange(len(labelled_traits.describe().columns.tolist()))
    plt.bar(X_axis - (width / 2), labelled_means, width=width, edgecolor='black', label='Labelled Data')
    # plt.bar(X_axis - width, kmm_corrected_means, width=width, edgecolor='black', label='KMM')
    # plt.bar(X_axis, simple_kmm_corrected_means, width=width, edgecolor='black', label='KMM Simple')
    # plt.bar(X_axis , ratio_corrected_means, width=width, edgecolor='black', label='Ratio')
    plt.bar(X_axis + (width / 2), logit_corrected_means, width=width, edgecolor='black', label='Logit')
    # plt.bar(X_axis + 2 * width, simple_ratio_corrected_means, width=width, edgecolor='black', label='Ratio Simple')
    plt.bar(X_axis + 1.5 * width, all_means, width=width, edgecolor='black', label='Underlying Pop.')

    plt.xticks(X_axis + width / 2, labelled_traits.describe().columns.tolist(), rotation=65)
    plt.legend()
    plt.xlabel('Trait')
    plt.ylabel('Scaled Mean')
    plt.tight_layout()
    plt.savefig(os.path.join(_comparison_output_dir, out_filename))
    plt.close()


def main():
    # Note this helps to verify the correction procedure but with limited labelled data it inevitably won't perfectly
    # match the underlying pop.
    disc_to_compare = [c for c in vars_to_use_in_bias_analysis if c in DISCRETE_VARS]
    plot_corrected_means(disc_to_compare, 'corrected_disc_means.png')
    cont_to_compare = [c for c in vars_to_use_in_bias_analysis if c in CONTINUOUS_VARS]
    plot_corrected_means(cont_to_compare, 'corrected_cont_means.png')
    # all_num_vars = [c for c in vars_to_use_in_bias_analysis if c in NUMERIC_TRAITS]
    # plot_corrected_means(all_num_vars, 'all_means.png')


if __name__ == '__main__':
    main()

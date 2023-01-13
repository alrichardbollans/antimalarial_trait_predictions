import os
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pkg_resources import resource_filename

from bias_correction_and_summaries import ALL_TRAITS, LABELLED_TRAITS, \
    vars_to_use_in_bias_analysis, WEIGHTED_LABELLED_DATA
from import_trait_data import CONTINUOUS_VARS, DISCRETE_VARS

_output_path = resource_filename(__name__, 'outputs')

_comparison_output_dir = os.path.join(_output_path, 'weighting_comparison')

if not os.path.isdir(_output_path):
    os.mkdir(_output_path)


def plot_corrected_means(vars_to_compare: List[str], out_filename: str, scale=True):
    from sklearn.preprocessing import MinMaxScaler
    weighted_labelled_df = pd.read_csv(WEIGHTED_LABELLED_DATA, index_col=0)
    corrected_df = weighted_labelled_df[vars_to_compare + ['weight']].copy(deep=True)

    all_traits = ALL_TRAITS[vars_to_compare]
    labelled_traits = LABELLED_TRAITS[vars_to_compare]
    ylab = 'Mean'
    if scale:
        # scale_features for plotting
        s = MinMaxScaler()
        s.fit(all_traits)
        all_traits = pd.DataFrame(s.transform(all_traits),
                                  columns=s.get_feature_names_out())
        labelled_traits = pd.DataFrame(s.transform(labelled_traits),
                                       columns=s.get_feature_names_out())

        scaled_corrected_df = pd.DataFrame(s.transform(corrected_df[vars_to_compare]),
                                           columns=s.get_feature_names_out())
        scaled_corrected_df['weight'] = corrected_df['weight'].values
        corrected_df = scaled_corrected_df
        ylab = 'Scaled Mean'

    labelled_means = labelled_traits.describe().loc[['mean']].values.tolist()[0]
    all_means = all_traits.describe().loc[['mean']].values.tolist()[0]

    logit_corrected_means = []
    for c in vars_to_compare:
        # Need to use a mask to ignore nans
        masked_data = np.ma.masked_array(corrected_df[c], np.isnan(corrected_df[c]))
        avg = np.average(masked_data,
                         weights=corrected_df['weight'])
        logit_corrected_means.append(avg)

    plt.figure(figsize=(8, 6))
    plt.rc('font', size=12)
    width = 0.2
    X_axis = np.arange(len(labelled_traits.describe().columns.tolist()))
    plt.bar(X_axis - (width / 2), labelled_means, width=width, edgecolor='black', label='Labelled Data')

    plt.bar(X_axis + (width / 2), logit_corrected_means, width=width, edgecolor='black',
            label='Corrected Data')
    plt.bar(X_axis + 1.5 * width, all_means, width=width, edgecolor='black', label='Underlying Pop.')

    plt.xticks(X_axis + width / 2, labelled_traits.describe().columns.tolist(), rotation=65)
    plt.legend()
    plt.xlabel('Trait')
    plt.ylabel(ylab)
    plt.tight_layout()
    plt.savefig(os.path.join(_comparison_output_dir, out_filename), dpi=400)
    plt.close()
    plt.cla()
    plt.clf()


def main():
    # Note this helps to verify the correction procedure but with limited labelled data it inevitably won't perfectly
    # match the underlying pop.
    disc_to_compare = [c for c in vars_to_use_in_bias_analysis if c in DISCRETE_VARS]
    plot_corrected_means(disc_to_compare, 'corrected_disc_means.jpg', scale=False)
    cont_to_compare = [c for c in vars_to_use_in_bias_analysis if c in CONTINUOUS_VARS]
    plot_corrected_means(cont_to_compare, 'corrected_cont_means.jpg')


if __name__ == '__main__':
    main()

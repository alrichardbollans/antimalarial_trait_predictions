import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from bias_correction_and_summaries import bias_output_dir, UNLABELLED_TRAITS, \
    LABELLED_TRAITS, logit_correction
from import_trait_data import TARGET_COLUMN

_pa_output_path = os.path.join(bias_output_dir, 'probability_analysis')
if not os.path.isdir(_pa_output_path):
    os.mkdir(_pa_output_path)

binwidth = 0.01


def weight_distribution():
    # sns.set()
    # sns.histplot(weighted_df['P(s|x)'], binwidth=binwidth)
    # plt.tight_layout()
    # plt.savefig(os.path.join(_pa_output_path, 'labelled_distribution.png'))
    # plt.close()
    sns.set()
    h = sns.histplot(weighted_unlabelled_df['P(s|x)'], binwidth=binwidth)
    h.set_xticks(np.linspace(0, 1, 11))
    plt.tight_layout()
    plt.savefig(os.path.join(_pa_output_path, 'unlabelled_distribution.png'))
    plt.close()


def probabilities_of_selecting_active_species():
    if binwidth == 0.01:
        bins = np.linspace(0, 1, 11)
    else:
        raise ValueError

    sns.set()
    groups = weighted_df[[TARGET_COLUMN, 'P(s|x)']].groupby(
        pd.cut(weighted_df['P(s|x)'], bins, include_lowest=True))
    props = groups[TARGET_COLUMN].mean()
    # props1 = groups[TARGET_COLUMN].sum()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.grid(False)
    #
    #
    sns.histplot(weighted_df, x='P(s|x)', bins=101, hue=TARGET_COLUMN, multiple='stack',ax=ax1)
    # plot_centers = (bins[:-1] + bins[1:]) / 2
    # sns.lineplot(x=plot_centers, y=props, ax=ax2, color='red', legend='brief')
    #
    # ax1.set_xticks(np.linspace(0, 1, 11))
    # ax2.set_ylabel('Mean Activity')


    plt.tight_layout()
    plt.savefig(os.path.join(_pa_output_path, 'labelled_distribution.png'))
    plt.close()


def main():
    weight_distribution()
    probabilities_of_selecting_active_species()


if __name__ == '__main__':
    weighted_df, weighted_unlabelled_df = logit_correction(LABELLED_TRAITS, UNLABELLED_TRAITS)
    main()

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from bias_correction_and_summaries import bias_output_dir, WEIGHTED_LABELLED_DATA, WEIGHTED_UNLABELLED_DATA
from import_trait_data import TARGET_COLUMN

_pa_output_path = os.path.join(bias_output_dir, 'probability_analysis')
if not os.path.isdir(_pa_output_path):
    os.mkdir(_pa_output_path)

binwidth = 0.01


def weight_distribution():
    weighted_unlabelled_df = pd.read_csv(WEIGHTED_UNLABELLED_DATA)

    sns.set(font_scale=1.2)

    h = sns.histplot(weighted_unlabelled_df['P(s|x)'], binwidth=binwidth)
    h.set_xticks(np.linspace(0, 1, 11))
    plt.tight_layout()
    plt.savefig(os.path.join(_pa_output_path, 'unlabelled_distribution.png'))
    plt.close()
    plt.cla()
    plt.clf()


def probabilities_of_selecting_active_species():
    weighted_labelled_df = pd.read_csv(WEIGHTED_LABELLED_DATA)
    if binwidth == 0.01:
        bins = 101
    else:
        raise ValueError

    sns.set(font_scale=1.2)

    h=sns.histplot(weighted_labelled_df, x='P(s|x)', bins=bins, hue=TARGET_COLUMN, multiple='stack')
    h.set_xticks(np.linspace(0, 1, 11))
    plt.tight_layout()
    plt.savefig(os.path.join(_pa_output_path,'P(s|x)_labelled_distribution.png'))
    plt.close()
    plt.cla()
    plt.clf()


def main():
    weight_distribution()
    probabilities_of_selecting_active_species()


if __name__ == '__main__':
    main()

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

binwidth = 0.001


def unlabelled_distribution():
    weighted_unlabelled_df = pd.read_csv(WEIGHTED_UNLABELLED_DATA)

    sns.set(font_scale=1.2)

    h = sns.histplot(weighted_unlabelled_df['P(s|x)'], binwidth=binwidth)
    h.set_xlim(0, 0.1)
    h.set_xticks(np.linspace(0, 0.1, 11))
    plt.tight_layout()
    plt.savefig(os.path.join(_pa_output_path, 'unlabelled_distribution.jpg'), dpi=400)
    plt.close()
    plt.cla()
    plt.clf()

    weighted_labelled_df = pd.read_csv(WEIGHTED_LABELLED_DATA)
    all_weighted_data = pd.concat([weighted_labelled_df, weighted_unlabelled_df])
    sns.set(font_scale=1.2)
    all_weighted_data[['P(s|x)']].describe().to_csv(os.path.join(_pa_output_path, 'summary.csv'))
    h = sns.histplot(all_weighted_data['P(s|x)'], binwidth=binwidth)
    h.set_xlim(0, 0.1)
    h.set_xticks(np.linspace(0, 0.1, 11))
    plt.tight_layout()
    plt.savefig(os.path.join(_pa_output_path, 'underlying_distribution.jpg'), dpi=400)
    plt.close()
    plt.cla()
    plt.clf()


def probabilities_of_selecting_active_species():
    weighted_labelled_df = pd.read_csv(WEIGHTED_LABELLED_DATA)

    bins = int((1 / 0.01) + 1)
    sns.set(font_scale=1.2)

    h = sns.histplot(weighted_labelled_df, x='P(s|x)', bins=bins, hue=TARGET_COLUMN, multiple='stack')
    h.set_xticks(np.linspace(0, 1, 11))
    plt.tight_layout()
    plt.savefig(os.path.join(_pa_output_path, 'P(s|x)_labelled_distribution.jpg'), dpi=400)
    plt.close()
    plt.cla()
    plt.clf()
    bins = int((1 / binwidth) + 1)
    h = sns.histplot(weighted_labelled_df, x='P(s|x)', bins=bins, hue=TARGET_COLUMN, multiple='stack')
    h.set_xlim(0, 0.1)
    h.set_xticks(np.linspace(0, 0.1, 11))
    plt.tight_layout()
    plt.savefig(os.path.join(_pa_output_path, 'P(s|x)_labelled_distribution_small.jpg'), dpi=400)
    plt.close()
    plt.cla()
    plt.clf()

    actives = weighted_labelled_df[weighted_labelled_df[TARGET_COLUMN] == 1]
    inactives = weighted_labelled_df[weighted_labelled_df[TARGET_COLUMN] == 0]
    pd.DataFrame(
        {'num': [actives['P(s|x)'].mean(), inactives['P(s|x)'].mean()]},
        index=['active', 'inactive']).to_csv(
        os.path.join(_pa_output_path, 'psmeans.csv'))


def main():
    unlabelled_distribution()
    probabilities_of_selecting_active_species()


if __name__ == '__main__':
    main()

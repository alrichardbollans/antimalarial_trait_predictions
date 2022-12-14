import os

import matplotlib.pyplot as plt
import seaborn as sns

from bias_correction_and_summaries import bias_output_dir, UNLABELLED_TRAITS, \
    LABELLED_TRAITS, logit_correction
from import_trait_data import TARGET_COLUMN

_pa_output_path = os.path.join(bias_output_dir, 'probability_analysis')
if not os.path.isdir(_pa_output_path):
    os.mkdir(_pa_output_path)


def weight_distribution():

    sns.set()
    sns.histplot(weighted_df['P(s)'], binwidth=0.05)
    plt.tight_layout()
    plt.savefig(os.path.join(_pa_output_path, 'labelled_distribution.png'))
    plt.close()
    sns.set()
    sns.histplot(weighted_unlabelled_df['P(s)'], binwidth=0.05)
    plt.tight_layout()
    plt.savefig(os.path.join(_pa_output_path, 'unlabelled_distribution.png'))
    plt.close()


def probabilities_of_selecting_active_species():

    sns.set()
    sns.histplot(weighted_df, x='P(s)', binwidth=0.05, hue=TARGET_COLUMN, multiple='stack')
    plt.tight_layout()
    plt.savefig(os.path.join(_pa_output_path, 'probabilities_of_selecting_active_species.png'))
    plt.close()


def main():

    weight_distribution()
    probabilities_of_selecting_active_species()



if __name__ == '__main__':
    weighted_df, weighted_unlabelled_df = logit_correction(LABELLED_TRAITS, UNLABELLED_TRAITS)
    main()

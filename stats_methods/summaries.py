import os
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from import_trait_data import TARGET_COLUMN, BINARY_VARS, HABIT_COLS, MORPH_VARS, TRAITS, NUMERIC_TRAITS


def activity_with_without_feature(labelled_df: pd.DataFrame, out_dir: str):
    means_with = []
    means_without = []
    for c in BINARY_VARS:
        means_with.append(labelled_df[labelled_df[c] == 1][TARGET_COLUMN].mean())
        means_without.append(labelled_df[labelled_df[c] == 0][TARGET_COLUMN].mean())
    plt.rcParams['figure.figsize'] = [10, 6]
    width = 0.33

    X_axis = np.arange(len(BINARY_VARS))
    plt.bar(X_axis, means_with, width=width, edgecolor='black', label='Presence')
    plt.bar(X_axis + width, means_without, width=width, edgecolor='black', label='Absence')

    plt.xticks(X_axis + width / 2, BINARY_VARS, rotation=65)
    plt.legend()
    plt.xlabel('Feature')
    plt.ylabel('Mean Activity')
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(out_dir, 'presence_absence_means.png'))
    plt.close()


def compare_active_vs_inactive(active_df: pd.DataFrame, inactive_df: pd.DataFrame, out_dir: str):
    active_df = active_df[TRAITS]
    inactive_df = inactive_df[TRAITS]
    active_means = active_df.describe().loc[['mean']].values.tolist()[0]
    inactive_means = inactive_df.describe().loc[['mean']].values.tolist()[0]
    plt.rcParams['figure.figsize'] = [10, 6]
    width = 0.33

    X_axis = np.arange(len(active_df.describe().columns.tolist()))
    plt.bar(X_axis, active_means, width=width, edgecolor='black', label='Active')
    plt.bar(X_axis + width, inactive_means, width=width, edgecolor='black', label='Inactive')

    plt.xticks(X_axis + width / 2, active_df.describe().columns.tolist(), rotation=65)
    plt.legend()
    plt.xlabel('Feature')
    plt.ylabel('Mean')
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(out_dir, 'activity_means.png'))
    plt.close()


def venn_diagram(df: pd.DataFrame, out_dir: str):
    from matplotlib_venn import venn3, venn3_circles
    from matplotlib import pyplot as plt
    active_df = df[df[TARGET_COLUMN] == 2]
    weak_df = df[df[TARGET_COLUMN] == 1]
    inactive_df = df[df[TARGET_COLUMN] == 0]
    # depict venn diagram
    a = len(active_df.index)
    w = len(weak_df.index)
    i = len(inactive_df.index)
    subsets = (0, 0, 0, i, 0, w, a)
    v = venn3(subsets=subsets, set_labels=('Active', 'Weak', 'Inactive'),
              set_colors=("red", "orange",
                          "grey"), alpha=0.8)
    for idx, subset in enumerate(v.subset_labels):
        if v.subset_labels[idx] is not None:
            v.subset_labels[idx].set_visible(False)
    for idx, subset in enumerate(v.set_labels):
        if v.set_labels[idx] is not None:
            v.set_labels[idx].set_visible(False)
    # add outline
    venn3_circles(subsets=subsets)
    plt.title('Activity')
    plt.savefig(os.path.join(out_dir, 'activity_venns.png'))
    # plt.show()
    plt.legend()
    plt.close()


# def plot_means_all_vs_labelled(dfs: List[pd.DataFrame]):
#     collection_of_means = []
#     for df in dfs:
#         collection_of_means.append(df.describe().loc[['mean']].values.tolist()[0])
#     # Remove features to not plot
#     habits = [c for c in _ALL_TRAITS.columns if 'habit_' in c]
#     all_traits = _ALL_TRAITS.drop(columns=[TARGET_COLUMN] + habits)
#     labelled_traits = LABELLED_TRAITS.drop(columns=[TARGET_COLUMN] + habits)
#     # Ethno Means
#     non_ethno_data = all_traits[all_traits['Medicinal'] == 0]
#     no_use_means = non_ethno_data.describe().loc[['mean']].values.tolist()[0]
#
#     ethno_data = all_traits[all_traits['Medicinal'] == 1]
#     history_of_use_means = ethno_data.describe().loc[['mean']].values.tolist()[0]
#
#     labelled_means = labelled_traits.describe().loc[['mean']].values.tolist()[0]
#     all_means = all_traits.describe().loc[['mean']].values.tolist()[0]
#
#     plt.rcParams['figure.figsize'] = [10, 6]
#     width = 0.2
#
#     X_axis = np.arange(len(labelled_traits.describe().columns.tolist()))
#
#     plt.bar(X_axis - width * 1.5, labelled_means, width=width, edgecolor='black', label='Labelled Data')
#     plt.bar(X_axis - width / 2, history_of_use_means, width=width, edgecolor='black', label='Medicinal Use')
#     plt.bar(X_axis + width / 2, no_use_means, width=width, edgecolor='black', label='Non-Medicinal Use')
#     plt.bar(X_axis + 1.5 * width, all_means, width=width, edgecolor='black', label='All Data')
#
#     plt.xticks(X_axis + width / 2, labelled_traits.describe().columns.tolist(), rotation=65)
#     plt.legend()
#     plt.xlabel('Feature')
#     plt.ylabel('Mean')
#     plt.tight_layout()
#     # plt.show()
#     plt.savefig(os.path.join(summary_output_dir, 'ethno_mean_comparison.png'))
#     plt.close()
def plot_means_all_vs_labelled(all_traits: pd.DataFrame, features_to_plot: List[str], output_file: str,
                               minmaxscale: bool = True):
    from matplotlib import pyplot as plt

    labelled_traits = all_traits[~all_traits[TARGET_COLUMN].isna()]

    all_traits = all_traits[features_to_plot]
    labelled_traits = labelled_traits[features_to_plot]
    ylabel = 'Mean'
    if minmaxscale:
        ylabel = 'Scaled Mean'
        # scale_features
        s = MinMaxScaler()
        s.fit(all_traits)
        all_traits = pd.DataFrame(s.transform(all_traits),
                                  columns=s.get_feature_names_out())
        labelled_traits = pd.DataFrame(s.transform(labelled_traits),
                                       columns=s.get_feature_names_out())

    X_axis = np.arange(len(labelled_traits.describe().columns.tolist()))
    # Plot given means
    labelled_means = labelled_traits.describe().loc[['mean']].values.tolist()[0]
    all_means = all_traits.describe().loc[['mean']].values.tolist()[0]
    # unlabelled_means = unlabelled_traits.describe().loc[['mean']].values.tolist()[0]

    plt.figure(figsize=(8, 5))
    width = 0.33
    plt.bar(X_axis, labelled_means, width=width, edgecolor='black', label='Labelled Sample')
    # plt.bar(X_axis + (width / 2), unlabelled_means, width=width, edgecolor='black', label='Unlabelled ')
    plt.bar(X_axis + (width), all_means, width=width, edgecolor='black', label='Underlying Pop.')

    plt.xticks(X_axis + width / 2, labelled_traits.describe().columns.tolist(), rotation=65)
    plt.legend(loc='upper right')
    plt.xlabel('Trait')
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

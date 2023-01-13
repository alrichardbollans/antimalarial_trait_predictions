import os.path
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pkg_resources import resource_filename

from bias_correction_and_summaries import vars_to_use_in_bias_analysis, ALL_TRAITS
from import_trait_data import TARGET_COLUMN, BINARY_VARS, HABIT_COLS

_output_path = resource_filename(__name__, 'outputs')

_comparison_output_dir = os.path.join(_output_path, 'heatmaps')


def get_cooccur_ratio_for_vars(df: pd.DataFrame, var1: str, var2: str):
    # Get relationship of variables i.e.
    # proportion of times variable 2 occurs when var1 occurs
    # Similar to precision/recall

    # Drop cases where one is unknown
    df = df.dropna(subset=[var1, var2], how='any')

    in_one_data = df[(df[var1] == 1)]
    in_both_data = df[(df[var1] == 1) & (df[var2] == 1)]

    try:
        value = float(len(in_both_data.index)) / len(in_one_data)
    except ZeroDivisionError:
        return 0

    if var1 == var2:
        assert value == 1
        return np.nan
    return round(value, 2)


def make_cooccur_df(df: pd.DataFrame, vars: List[str]):
    ## Make df
    df['All Species'] = 1
    var1_list = []
    var2_list = []
    values = []
    for p1 in vars + ['All Species']:
        for p2 in vars:
            var1_list.append(p1)
            var2_list.append(p2)
            values.append(get_cooccur_ratio_for_vars(df, p1, p2))

    out_data = pd.DataFrame({'Var1': var1_list, 'Var2': var2_list, 'Values': values})

    return out_data


def make_heatmap(cooccur_df: pd.DataFrame, outfile: str):
    plt.rc('font', size=15)  # controls default text size

    result = cooccur_df.pivot(index='Var2', columns='Var1', values='Values')

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        result,
        cmap="coolwarm",
        # vmax=1,
        annot=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.7},
        ax=ax,
    )
    plt.xlabel('')
    plt.ylabel('')

    plt.yticks(rotation=0)

    f.tight_layout()
    plt.savefig(outfile, dpi=400)
    plt.close()
    plt.cla()
    plt.clf()


def main():
    bin_traits = [c for c in vars_to_use_in_bias_analysis if c in BINARY_VARS]
    co_occ_df = make_cooccur_df(ALL_TRAITS, bin_traits + [TARGET_COLUMN])
    make_heatmap(co_occ_df, os.path.join(_comparison_output_dir, 'bin_heatmap.jpg'))


if __name__ == '__main__':
    main()

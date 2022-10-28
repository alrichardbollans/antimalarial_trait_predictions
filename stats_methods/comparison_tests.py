from typing import List

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, epps_singleton_2samp, ks_2samp, mannwhitneyu

from import_trait_data import DISCRETE_VARS, TARGET_COLUMN, BINARY_VARS, CONTINUOUS_VARS


def compare_samples_discrete_vars(df1: pd.DataFrame, df2: pd.DataFrame, vars_to_check: List[str] = None,
                                  output_csv: str = None):
    """
    Calculate statistics to compare datasets.
    Null hypothesis that the datasets are the same
    """

    all_results = []

    if vars_to_check is None:
        features = DISCRETE_VARS + [TARGET_COLUMN]
    else:
        features = vars_to_check

    for c in features:
        if c in df1.columns and c in df2.columns:
            if c in DISCRETE_VARS + [TARGET_COLUMN]:
                positive_in_1 = len(df1[df1[c] == 1][c].values)
                negative_in_1 = len(df1[df1[c] == 0][c].values)

                positive_in_2 = len(df2[df2[c] == 1][c].values)
                negative_in_2 = len(df2[df2[c] == 0][c].values)

                cont_table = np.array([[positive_in_1, negative_in_1], [positive_in_2, negative_in_2]])
                try:
                    chi2, p, dof, ex = chi2_contingency(cont_table)

                    chi = (chi2, p)
                except ValueError:
                    chi = (np.nan, np.nan)

                fisher = fisher_exact(cont_table)
            else:
                chi = (np.nan, np.nan)
                fisher = (np.nan, np.nan)

            x = df1[c].values
            y = df2[c].values
            if c not in BINARY_VARS:

                mur = mannwhitneyu(x, y, nan_policy='omit')

            else:
                mur = (np.nan, np.nan)
            all_results.append([c] + list(mur) + list(fisher) + list(chi))

    d = pd.DataFrame(all_results)

    d.set_axis(
        ['Feature', 'mwu_Statistic', 'mwu_p',
         'fisher_odds_ratio', 'fisher_p', 'chi2_stat', 'chi2_p'],
        axis=1, inplace=True)

    if output_csv is not None:
        d.to_csv(output_csv)
    return d


def compare_samples_continuous_vars(df1: pd.DataFrame, df2: pd.DataFrame, output_csv: str = None):
    """
    Calculate statistics to compare datasets.
    Null hypothesis that the datasets are the same
    """

    all_results = []
    features = CONTINUOUS_VARS
    for c in features:
        if c in df1.columns and c in df2.columns:
            x = df1[c].dropna().values
            y = df2[c].dropna().values

            ksr = ks_2samp(x, y)

            mur = mannwhitneyu(x, y, nan_policy='omit')

            all_results.append([c] + list(ksr) + list(mur))

    if len(all_results) > 0:
        d = pd.DataFrame(all_results)
        d.set_axis(
            ['Feature', 'ks_Statistic', 'ks_p_value', 'mwu_Statistic', 'mwu_p_value'
             ],
            axis=1, inplace=True)
    else:
        d = pd.DataFrame()

    if output_csv is not None:
        d.to_csv(output_csv)
    return d


def compare_sample_distributions(df1: pd.DataFrame, df2: pd.DataFrame, output_csv: str = None) -> pd.DataFrame:
    d = compare_samples_discrete_vars(df1, df2, vars_to_check=DISCRETE_VARS)
    c = compare_samples_continuous_vars(df1, df2)
    if len(c.index) > 0:
        out = pd.merge(c, d, how='outer')
    else:
        out = d
    if output_csv is not None:
        out.to_csv(output_csv)
    return out


def holm_bonferroni_correction(df: pd.DataFrame, p_value_col: str):
    # Holm-Bonferroni Method
    # Sture Holm, ‘A Simple Sequentially Rejective Multiple Test Procedure’, Scandinavian Journal of Statistics 6, no. 2 (1979): 65–70.

    new_df = df.sort_values(by=p_value_col)
    n = len(new_df.index)
    new_df.reset_index(inplace=True, drop=True)

    new_df['corrected_p_value'] = new_df.apply(lambda x: x[p_value_col] * (n - x.name), axis=1)

    return new_df

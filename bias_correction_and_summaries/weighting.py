import os
from typing import List

import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
from pkg_resources import resource_filename
from sklearn.linear_model import LogisticRegression

from bias_correction_and_summaries import LABELLED_TRAITS, UNLABELLED_TRAITS
from import_trait_data import CONTINUOUS_VARS

bias_output_dir = resource_filename(__name__, 'outputs')
_temp_output_dir = resource_filename(__name__, 'temp_outputs')

# num_features = [x for x in NUMERIC_TRAITS if x in vars_without_target_to_use] + ['Family']

# Choice motivated in discussion
known_biasing_features = ['Antimalarial_Use',
                          'Tested_for_Alkaloids',
                          'Medicinal',
                          'In_Malarial_Region', 'Genus', 'Family']
to_target_encode = ['Genus', 'Family']


def normalise_weight_column(df: pd.DataFrame):
    max_val = df['weight'].max()
    df['weight'] = df['weight'] / max_val
    return df


def kmm_correction(sample, underlying_pop_df, selection_vars, weight_bound=1000, no_features=None,
                   kernel_type='rbf') -> pd.DataFrame:
    from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel

    sample_copy = sample.copy(deep=True)
    underlying_pop_df_copy = underlying_pop_df.copy(deep=True)

    sample_to_use = sample_copy[selection_vars]

    # Check no nans in sample
    nan_samples = sample_to_use[sample_to_use.isna().any(axis=1)]
    if len(nan_samples.index) != 0:
        raise ValueError('Nan values in sample data')

    underlying_pop_to_use = underlying_pop_df_copy[selection_vars]

    nan_pops = underlying_pop_to_use[underlying_pop_to_use.isna().any(axis=1)]
    if len(nan_pops.index) != 0:

        nan_cols = []
        for c in nan_pops.columns:
            c_nan = nan_pops[nan_pops[c].isna()]
            if len(c_nan.index) != 0:
                nan_cols.append(c)
        print(f'Warning NaNs in underlying population data: filling NaNs in cols {nan_cols}')

        underlying_pop_to_use.fillna(0, inplace=True)

    # From  Huang, Jiayuan, Arthur Gretton, Karsten M Borgwardt, Bernhard Schölkopf, and Alex J Smola. ‘Correcting Sample Selection Bias by Unlabeled Data’, 2006.
    number_samples = len(sample_to_use.index)
    num_test_cases = len(underlying_pop_to_use.index)
    # This choice of epsilon given on p.5
    epsilon = float(weight_bound) / np.sqrt(number_samples)
    # epsilon = float(np.sqrt(number_samples) - 1) / np.sqrt(number_samples)

    if kernel_type == 'poly':
        if no_features is None:
            # When overfitting is not an issue, can use equal number of dimensions
            no_features = len(sample_to_use.columns)
        # Kernel must be universal
        K = polynomial_kernel(np.asarray(sample_to_use), degree=no_features)
        kappa = np.sum(
            polynomial_kernel(np.asarray(sample_to_use), np.asarray(underlying_pop_to_use), degree=no_features) * float(
                number_samples) / float(
                num_test_cases), axis=1)

    elif kernel_type == 'rbf':
        gam = 10
        K = rbf_kernel(np.asarray(sample_to_use), gamma=gam)
        kappa = np.sum(rbf_kernel(np.asarray(sample_to_use), np.asarray(underlying_pop_to_use), gamma=gam) * float(
            number_samples) / float(
            num_test_cases), axis=1)
    else:
        raise ValueError('Unknown Kernel method')
    K = matrix(K)
    kappa = matrix(kappa)
    G = matrix(np.r_[np.ones((1, number_samples)), -np.ones((1, number_samples)), np.eye(number_samples), -np.eye(
        number_samples)])
    h = matrix(
        np.r_[number_samples * (1 + epsilon), number_samples * (epsilon - 1), 1 * np.ones((number_samples,)), np.zeros(
            (number_samples,))])

    sol = solvers.qp(K, -kappa, G, h)
    weights = np.array(sol['x'])

    out = sample.copy()
    out['weight'] = weights

    return out


def ratio_correction(trait_df, underlying_pop_df, selection_vars, max_weight=None) -> pd.DataFrame:
    # Get different combinations of biasing variables and how often they occur in biased data and all data

    trait_df_copy = trait_df.copy(deep=True)
    underlying_pop_df_copy = underlying_pop_df.copy(deep=True)

    # Avoid editing list
    selection_vars_copy = selection_vars.copy()
    # bucket continuous vars
    cols_to_drop_at_end = []

    all_data_for_binning = pd.concat([underlying_pop_df_copy, trait_df_copy])
    for c in CONTINUOUS_VARS:
        if c in selection_vars_copy:
            binned_c = c + '_binned'
            # Generate bins from all data

            all_data_for_binning[binned_c], bins = pd.cut(all_data_for_binning[c], 2, labels=[0, 1], retbins=True)
            underlying_pop_df_copy[binned_c] = pd.cut(underlying_pop_df_copy[c], bins, labels=[0, 1])
            trait_df_copy[binned_c] = pd.cut(trait_df_copy[c], bins, labels=[0, 1])

            underlying_pop_df_copy[binned_c] = underlying_pop_df_copy[binned_c].astype(int)
            trait_df_copy[binned_c] = trait_df_copy[binned_c].astype(int)
            selection_vars_copy.remove(c)
            selection_vars_copy.append(binned_c)
            cols_to_drop_at_end.append(binned_c)
    biased_data_combinations = trait_df_copy.groupby(selection_vars_copy, as_index=False,
                                                     dropna=False).size()
    all_data_combinations = underlying_pop_df_copy.groupby(selection_vars_copy, as_index=False, dropna=False).size()

    # Merge on these biasing variables to add n and m columns
    m = trait_df_copy.merge(all_data_combinations, on=selection_vars_copy, how='left')
    m.rename(columns={'size': 'n'}, inplace=True)
    out = m.merge(biased_data_combinations, on=selection_vars_copy, how='left')
    out.rename(columns={'size': 'm'}, inplace=True)
    out['n'] = out['n'].fillna(0)
    out['weight'] = out['n'] / out['m']
    # if allow_data_leak:
    #     m_weight = out['weight'].max()
    #     min_weight = out['weight'].min()
    #     if m_weight < 1 or min_weight < 1:
    #         raise ValueError

    if max_weight is not None:
        out['weight'] = out['weight'].clip(upper=max_weight)

    out.drop(columns=['m', 'n'] + cols_to_drop_at_end, inplace=True)

    return out


def logit_correction(trait_df: pd.DataFrame, underlying_pop_df: pd.DataFrame, selection_vars: List[str],
                     cols_to_target_encode: List[str]) -> pd.DataFrame:
    selected = trait_df.copy(deep=True)
    population_copy = underlying_pop_df.copy(deep=True)[selection_vars]

    all_data = pd.concat([selected, population_copy])
    dup_df1 = all_data[all_data.duplicated()]
    if len(dup_df1.index) > 0:
        print('Warning: repeated elements in sample and underlying pop. for Logit correction')
        print('Copies appearing in sample should be removed from underlying pop.')
        print(dup_df1)
        dup_df1.to_csv(os.path.join(_temp_output_dir, 'logit_dups.csv'))

    selected['selected'] = 1
    population_copy['selected'] = 0

    train = pd.concat([selected, population_copy])

    train_X = train[selection_vars]
    train_Y = train['selected']

    if cols_to_target_encode is not None:
        for c in cols_to_target_encode:
            # If possible to convert to float, raise error as these should be strings
            try:
                test1 = train[c].astype(float)
            except ValueError:
                pass
            else:
                raise ValueError(f'Trying to target encode floats: {c}. Have they already been encoded?')
        import category_encoders as ce
        # Target encode genus and family for logit correction:
        target_encoder = ce.TargetEncoder(cols=cols_to_target_encode)
        target_encoder.fit(train_X, train_Y)
        encoded_X_train = target_encoder.transform(train_X)
        encoded_selected = target_encoder.transform(selected[selection_vars])
    else:
        encoded_X_train = train_X
        encoded_selected = selected[selection_vars]

    logit = LogisticRegression()

    logit.fit(encoded_X_train, train_Y)

    prob_estimates = logit.predict_proba(encoded_selected)[:, 1]
    selected['weight'] = prob_estimates
    selected['weight'] = 1 / selected['weight']
    selected.drop(columns=['selected'], inplace=True)
    pd.testing.assert_index_equal(selected.index, trait_df.index)
    return selected


def append_weight_column(trait_df: pd.DataFrame, underlying_pop_df: pd.DataFrame, method: str,
                         selection_vars: List[str]) -> pd.DataFrame:
    """
    The ratio procedure of weighting follows Cortes, Corinna, Mehryar Mohri, Michael Riley, and Afshin Rostamizadeh. ‘Sample Selection Bias Correction Theory’. In Algorithmic Learning Theory, edited by Yoav Freund, László Györfi, György Turán, and Thomas Zeugmann, 5254:38–53. Lecture Notes in Computer Science. Berlin, Heidelberg: Springer Berlin Heidelberg, 2008. https://doi.org/10.1007/978-3-540-87987-9_8.
    Note, unless allowed, to avoid data leakage we use trait_df here with Unlabelled traits, rather than Labelled traits and all_traits
    :param trait_df:
    :param method: 'ratio' or 'kmm'
    :param selection_vars:
    :param allow_data_leak:
    :return:
    """

    print(f'Selection vars for bias correction: {selection_vars}')
    if method == 'ratio':
        out = ratio_correction(trait_df, underlying_pop_df, selection_vars)

    elif method == 'kmm':

        out = kmm_correction(trait_df, underlying_pop_df, selection_vars)
    else:
        raise ValueError('Unrecognised method for bias correction')

    assert 'weight' not in trait_df.columns
    assert 'weight' not in underlying_pop_df.columns

    return out


def oversample_by_weight(trait_df: pd.DataFrame, underlying_pop_df: pd.DataFrame, method: str,
                         selection_vars, cols_to_target_encode=None) -> pd.DataFrame:
    if method == 'logit':
        weight_df = logit_correction(trait_df, underlying_pop_df, selection_vars=selection_vars,
                                     cols_to_target_encode=cols_to_target_encode)
    else:
        weight_df = append_weight_column(trait_df, underlying_pop_df, method=method, selection_vars=selection_vars)

    min_weight = weight_df['weight'].min()
    if min_weight < 1:
        # Can't always divide by min_weight as this creates enormous arrays

        weight_df['weight'] = weight_df['weight'] / max(min_weight, 0.0001)

    over_df = weight_df.loc[weight_df.index.repeat(weight_df['weight'])]

    w_sum = weight_df['weight'].astype(int).sum()
    if w_sum != len(over_df.index):
        raise ValueError(f'{w_sum}:{len(over_df.index)}')

    over_df.drop(columns=['weight'], inplace=True)
    return over_df


def main():
    logit_correction(LABELLED_TRAITS, UNLABELLED_TRAITS,
                     known_biasing_features, to_target_encode).to_csv(
        os.path.join(bias_output_dir, 'weigthed_data_logit.csv'))


if __name__ == '__main__':
    main()

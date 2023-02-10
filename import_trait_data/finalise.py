import os

import pandas as pd
from typing import List

from import_trait_data import ACCEPTED_NAME_COLUMN, TARGET_COLUMN, COLUMNS_TO_DROP, NAME_COLUMNS, \
    TEMP_ALL_TRAITS_CSV, \
    TRAITS_WITHOUT_NANS, GENERA_VARS, TRAITS, TAXONOMIC_VARS, IMPORTED_TRAIT_CSV, \
    TRAITS_TO_DROP_AFTER_IMPORT, IMPORTED_LABELLED_TRAIT_CSV, IMPORTED_UNLABELLED_TRAIT_CSV, IMPORT_OUTPUT_DIR


def remove_samples_with_no_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes samples which don't have any given data (except taxanomic data)
    :param df:
    :return:
    """
    non_taxa_traits = [x for x in TRAITS if x not in TAXONOMIC_VARS] + [TARGET_COLUMN]
    mask = ' & '.join(
        [f'(df["{c}"].isna())' for c in
         non_taxa_traits])
    num_nans_removed = len(df[eval(mask)])
    print(f'{num_nans_removed} samples have been removed due to lack of data')
    df = df.dropna(subset=non_taxa_traits, how='all')
    return df


def replace_nans_with_zeros(df: pd.DataFrame):
    for col in TRAITS_WITHOUT_NANS:
        df[col] = df[col].fillna(0)


def fill_in_genera_vars(in_df: pd.DataFrame):
    """
    For traits that are considered to be at the genera level, fills in taxa where some information is given at the genera level
    Note that 'positive' hits are given priority
    :param in_df:
    :return:
    """
    for var in GENERA_VARS:
        if var in in_df.columns:
            genera_hits = in_df[in_df[var] == 1]['Genus'].dropna().unique()
            onemask = in_df[var].isna() & in_df['Genus'].isin(genera_hits)
            onemask_df = in_df[onemask]
            in_df.loc[onemask, var] = 1

            genera_zeros = in_df[in_df[var] == 0]['Genus'].dropna().unique()
            zeromask = in_df[var].isna() & in_df['Genus'].isin(genera_zeros)
            zeromaskmask_df = in_df[zeromask]
            in_df.loc[zeromask, var] = 0
        else:
            print(f'Warning: {var} specified in variables but not in Dataframe')


def fill_in_rubiaceae_latex(in_df: pd.DataFrame):
    """
    In general, Rubiaceae is known to not contain latex
    :param in_df:
    :return:
    """

    zeromask = in_df['Coloured_Latex'].isna() & (in_df['Family'] == 'Rubiaceae')
    zeromaskmask_df = in_df[zeromask]
    in_df.loc[zeromask, 'Coloured_Latex'] = 0


def get_genus_species_list(lab_df: pd.DataFrame, unlab_df: pd.DataFrame):
    def pad_dict_list(dict_list, padel):
        lmax = 0
        for lname in dict_list.keys():
            lmax = max(lmax, len(dict_list[lname]))
        for lname in dict_list.keys():
            ll = len(dict_list[lname])
            if ll < lmax:
                dict_list[lname] += [padel] * (lmax - ll)
        return dict_list


def tidy_var_names(names:List[str]):
    ### For some outputs, clean list of given names
    return [c.replace('_', ' ') for c in names]


def standardise_output(df: pd.DataFrame, out_csv: str = None):
    df = remove_samples_with_no_data(df)
    replace_nans_with_zeros(df)

    df = df[NAME_COLUMNS +
            [c for c in df if
             (c not in [TARGET_COLUMN] and c not in NAME_COLUMNS and c not in TRAITS_TO_DROP_AFTER_IMPORT)]
            + [TARGET_COLUMN]]

    if out_csv is not None:
        df.to_csv(out_csv)

    return df


def main():
    # Read with accepted name as index
    trait_df = pd.read_csv(TEMP_ALL_TRAITS_CSV, index_col=0)

    # Remove unspecified species
    trait_df.dropna(subset=[ACCEPTED_NAME_COLUMN], inplace=True)

    # Drop unwanted columns
    try:
        trait_df.drop(columns=COLUMNS_TO_DROP, inplace=True)
    except KeyError as e:
        raise KeyError(f'{e}: {trait_df.columns}')
    trait_df = trait_df.astype(
        dtype={'Family': "str"})
    x = trait_df.copy(deep=True)
    fill_in_genera_vars(trait_df)
    if len(x.compare(trait_df).index) > 0:
        print(x.compare(trait_df))
        print(
            'Variables should only appear in this comparison when they have been given in the trait table at species level but updated at genera level')

    all_taxa = trait_df.copy(deep=True)
    all_taxa = standardise_output(all_taxa, IMPORTED_TRAIT_CSV)
    labelled_taxa = all_taxa[~(all_taxa[TARGET_COLUMN].isna())]
    labelled_taxa.to_csv(IMPORTED_LABELLED_TRAIT_CSV)
    unlabelled_taxa = all_taxa[all_taxa[TARGET_COLUMN].isna()]
    unlabelled_taxa.to_csv(IMPORTED_UNLABELLED_TRAIT_CSV)

    labelled_taxa.describe().to_csv(os.path.join(IMPORT_OUTPUT_DIR, 'labelled trait summary.csv'))
    unlabelled_taxa.describe().to_csv(
        os.path.join(IMPORT_OUTPUT_DIR, 'unlabelled trait summary.csv'))
    all_taxa.describe().to_csv(os.path.join(IMPORT_OUTPUT_DIR, 'all trait summary.csv'))

    log_all_taxa = all_taxa[all_taxa['Family'] == 'Loganiaceae']
    log_all_taxa.describe().to_csv(os.path.join(IMPORT_OUTPUT_DIR, 'loganiaceae trait summary.csv'))

    log_all_taxa = all_taxa[all_taxa['Family'] == 'Apocynaceae']
    log_all_taxa.describe().to_csv(os.path.join(IMPORT_OUTPUT_DIR, 'Apocynaceae trait summary.csv'))

    log_all_taxa = all_taxa[all_taxa['Family'] == 'Rubiaceae']
    log_all_taxa.describe().to_csv(os.path.join(IMPORT_OUTPUT_DIR, 'Rubiaceae trait summary.csv'))


if __name__ == '__main__':
    main()

import pandas as pd

from import_trait_data import ACCEPTED_NAME_COLUMN, TARGET_COLUMN, COLUMNS_TO_DROP, NAME_COLUMNS, GENERA_LIST_CSV, \
    SPECIES_LIST_CSV, TEMP_ALL_TRAITS_CSV, FINAL_TRAITS_CSV, LABELLED_TRAITS_CSV, UNLABELLED_TRAITS_CSV, \
    TRAITS_WITHOUT_NANS, GENERA_VARS, TRAITS, TAXONOMIC_VARS, NON_MALARIAL_TRAITS_CSV, TAXA_IN_ALL_REGIONS_CSV, \
    TEMP_ALKALOID_CLASS_DATA_CSV, TAXA_TESTED_FOR_ALK_CLASSES_IN_ALL_REGIONS_CSV, NON_MALARIAL_LABELLED_TRAITS_CSV


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
        genera_hits = in_df[in_df[var] == 1]['Genus'].dropna().unique()
        onemask = in_df[var].isna() & in_df['Genus'].isin(genera_hits)
        onemask_df = in_df[onemask]
        in_df.loc[onemask, var] = 1

        genera_zeros = in_df[in_df[var] == 0]['Genus'].dropna().unique()
        zeromask = in_df[var].isna() & in_df['Genus'].isin(genera_zeros)
        zeromaskmask_df = in_df[zeromask]
        in_df.loc[zeromask, var] = 0


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

    lab_genera = list(lab_df['Genus'].unique())
    unlab_genera = list(unlab_df['Genus'].unique())
    out = {'Labelled': lab_genera, 'Unlabelled': unlab_genera}
    out = pad_dict_list(out, '')
    gen_df = pd.DataFrame(out)
    gen_df.to_csv(GENERA_LIST_CSV)

    lab_sps = list(lab_df['Accepted_Species'].unique())
    unlab_sps = list(unlab_df['Accepted_Species'].unique())
    out = {'Labelled': lab_sps, 'Unlabelled': unlab_sps}
    out = pad_dict_list(out, '')
    sp_df = pd.DataFrame(out)
    sp_df.to_csv(SPECIES_LIST_CSV)


def standardise_output(df: pd.DataFrame, out_csv: str = None):
    df = remove_samples_with_no_data(df)
    replace_nans_with_zeros(df)

    df = df[NAME_COLUMNS +
            [c for c in df if (c not in [TARGET_COLUMN] and c not in NAME_COLUMNS)]
            + [TARGET_COLUMN]]
    df.set_index(ACCEPTED_NAME_COLUMN)
    if out_csv is not None:
        df.to_csv(out_csv)

    return df


def get_subset_of_data_tested_for_alkaloid_classes(df: pd.DataFrame):
    alk_class_temp_df = pd.read_csv(TEMP_ALKALOID_CLASS_DATA_CSV)
    tested_species = alk_class_temp_df['Accepted_Species'].unique()
    tested_df = df[df['Accepted_Species'].isin(tested_species)]

    return tested_df


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
    # fill_in_rubiaceae_latex(trait_df)
    if len(x.compare(trait_df).index) > 0:
        print(x.compare(trait_df))
        print(
            'Variables should only appear in this comparison when they have been given in the trait table at species level but updated at genera level')

    # Remove taxa not in malarial regions
    non_malarial_taxa = trait_df[~(trait_df['In_Malarial_Region'] == 1)]
    non_malarial_taxa = standardise_output(non_malarial_taxa, NON_MALARIAL_TRAITS_CSV)
    print('Number labelled taxa not found in malarial regions: ')
    non_malarial_labelled = non_malarial_taxa[~non_malarial_taxa['Activity_Antimalarial'].isna()]
    print(len(non_malarial_labelled.index))
    non_malarial_labelled.to_csv(NON_MALARIAL_LABELLED_TRAITS_CSV)

    taxa_in_all_regions = trait_df.copy(deep=True)
    taxa_in_all_regions = standardise_output(taxa_in_all_regions)
    taxa_in_all_regions.to_csv(TAXA_IN_ALL_REGIONS_CSV)

    tested_alk_data = get_subset_of_data_tested_for_alkaloid_classes(taxa_in_all_regions)
    tested_alk_data.to_csv(TAXA_TESTED_FOR_ALK_CLASSES_IN_ALL_REGIONS_CSV)

    trait_df = trait_df[trait_df['In_Malarial_Region'] == 1]
    trait_df = standardise_output(trait_df)

    # Remove unspecified activity values
    labelled_trait_df = trait_df.drop(trait_df[trait_df[TARGET_COLUMN].isna()].index,
                                      inplace=False)
    labelled_trait_df = labelled_trait_df.astype(dtype={TARGET_COLUMN: "int64"})

    unlabelled_trait_df = trait_df.drop(trait_df[~trait_df[TARGET_COLUMN].isna()].index, inplace=False)
    unlabelled_trait_df = unlabelled_trait_df.drop(columns=[TARGET_COLUMN],
                                                   inplace=False)

    get_genus_species_list(labelled_trait_df, unlabelled_trait_df)

    trait_df.to_csv(FINAL_TRAITS_CSV)
    labelled_trait_df.to_csv(LABELLED_TRAITS_CSV)
    unlabelled_trait_df.to_csv(UNLABELLED_TRAITS_CSV)


if __name__ == '__main__':
    main()

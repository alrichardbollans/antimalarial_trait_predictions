import numpy as np
import pandas as pd
# Add progress bar to apply method
from getting_logan_malarial_regions import logan_taxa_in_malarial_countries_csv
from logan_climate_vars import logan_compiled_climate_vars_csv
from logan_common_name_vars import output_logan_common_names_csv
from logan_manually_collected_data import logan_encoded_traits_csv
from logan_medicinal_usage_vars import output_logan_medicinal_csv, output_logan_malarial_csv
from logan_metabolite_vars import logan_antibac_metabolite_hits_output_csv, logan_alkaloid_hits_output_csv, \
    logan_steroid_hits_output_csv, logan_cardenolide_hits_output_csv
from logan_morphological_vars import logan_habits_output_csv, logan_spines_output_csv, logan_no_spines_output_csv, \
    logan_hairy_output_csv
from logan_wcsp_distributions import logan_distributions_csv
from logan_wikipedia_vars import output_logan_wiki_csv
from manually_collected_data import encoded_traits_csv
from tqdm import tqdm
from wcsp_distributions import distributions_csv

tqdm.pandas()
from typing import List

from automatchnames import COL_NAMES, get_genus_from_full_name

from climate_vars import compiled_climate_vars_csv
from common_name_vars import output_common_names_csv
from import_trait_data import FAMILIES_OF_INTEREST, FAMILY_TAXA_CSV, TRAIT_CSV_WITH_EXTRA_TAXA, \
    TEMP_ALL_TRAITS_CSV, \
    ENVIRON_VARS, GENERA_VARS, DISCRETE_VARS, NON_NUMERIC_TRAITS, CONTINUOUS_VARS, TARGET_COLUMN
from getting_malarial_regions import taxa_in_malarial_countries_csv
from medicinal_usage_vars import output_medicinal_csv, output_malarial_csv
from morphological_vars import spines_output_csv, hairy_output_csv, no_spines_output_csv, habits_output_csv
from poison_vars import output_poison_csv, output_nonpoison_csv
from taxa_lists import get_all_taxa
from wikipedia_vars import output_wiki_csv
from metabolite_vars import rub_apoc_antibac_metabolite_hits_output_csv, rub_apoc_cardenolide_hits_output_csv, \
    rub_apoc_steroid_hits_output_csv, rub_apoc_alkaloid_hits_output_csv

variable_renaming = {}


def append_wcvp_taxa_to_trait_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds taxa (species, subspecies, varieties) from FAMILIES_OF_INTEREST
    :param df:
    :return:
    """
    # Load list of accepted taxa
    accepted_taxa = get_all_taxa(families_of_interest=FAMILIES_OF_INTEREST, ranks=["Species"],
                                 accepted=True)

    # Remove samples which already appear in our data
    accepted_samples_not_in_trait_df = accepted_taxa[
        ~accepted_taxa['kew_id'].isin(df['Accepted_ID'].values)].copy()

    accepted_samples_not_in_trait_df.rename(
        columns={'family': 'Family', 'genus': 'Genus', 'species': 'Species', 'kew_id': 'Accepted_ID',
                 'taxon_name': COL_NAMES['acc_name'], 'rank': COL_NAMES['acc_rank']}, inplace=True)
    accepted_samples_not_in_trait_df['Name'] = accepted_samples_not_in_trait_df[COL_NAMES['acc_name']]

    # Add accepted species column
    accepted_samples_not_in_trait_df[COL_NAMES['acc_species']] = accepted_samples_not_in_trait_df[
                                                                     'Genus'] + " " + \
                                                                 accepted_samples_not_in_trait_df['Species']

    # Add accepted species ids
    accepted_samples_not_in_trait_df['Accepted_Species_ID'] = np.nan
    accepted_samples_not_in_trait_df.loc[accepted_samples_not_in_trait_df[COL_NAMES['acc_rank']] == 'Species',
                                         'Accepted_Species_ID'] = accepted_samples_not_in_trait_df['Accepted_ID']
    accepted_samples_not_in_trait_df.loc[
        accepted_samples_not_in_trait_df['parent_name'] == accepted_samples_not_in_trait_df[COL_NAMES['acc_species']],
        'Accepted_Species_ID'] = accepted_samples_not_in_trait_df['parent_kew_id']

    if len(accepted_samples_not_in_trait_df[accepted_samples_not_in_trait_df['Accepted_Species_ID'].isna()].index) > 0:
        raise ValueError(
            f'Unassigned accepted species ids: {accepted_samples_not_in_trait_df[accepted_samples_not_in_trait_df["Accepted_Species_ID"]]}')

    cols_to_drop = [c for c in accepted_samples_not_in_trait_df if c not in df]

    accepted_samples_not_in_trait_df.drop(columns=cols_to_drop, inplace=True)

    cols_to_add = [c for c in df if c not in accepted_samples_not_in_trait_df]
    for c in cols_to_add:
        accepted_samples_not_in_trait_df[c] = np.nan
    accepted_samples_not_in_trait_df.to_csv(FAMILY_TAXA_CSV)

    updated_trait_df = pd.concat([df, accepted_samples_not_in_trait_df])

    updated_trait_df.to_csv(TRAIT_CSV_WITH_EXTRA_TAXA)

    return updated_trait_df


def encode_habits(df: pd.DataFrame) -> pd.DataFrame:
    # OHE habits
    def convert_habits_to_lists(hab: str) -> List[str]:

        if hab == '?':
            return ['unknown']
        try:
            if '/' not in hab:
                return [hab]
            else:
                return hab.split('/')
        except TypeError:
            if not np.isnan(hab):
                raise ValueError
            return ['unknown']

    df['Habit'] = df['Habit'].apply(convert_habits_to_lists)


    multilabels = df.Habit.str.join('|').str.get_dummies()
    df = df.join(multilabels)
    df.drop(columns=['Habit'], inplace=True)
    # Where habit is unknown, set all ohencoded columns to nan
    new_habit_cols = ['herb', 'liana', 'succulent', 'shrub', 'subshrub', 'tree']
    for h in new_habit_cols:
        if h != 'unknown':
            df.loc[df['unknown'] == 1, h] = np.nan
    df.drop(columns=['unknown'], inplace=True)
    return df


def read_csv_list(var_csvs: List[str]):
    var_dfs = []
    for filename in var_csvs:
        df = pd.read_csv(filename, index_col=0)
        var_dfs.append(df)

    var_df = pd.concat(var_dfs, axis=0, ignore_index=True)
    return var_df


def attach_new_var(in_df: pd.DataFrame, var_name: str, var_csvs: List[str], value_to_assign=1, level=None):
    if level is None:
        print('Must provide level!')
        raise ValueError
    if var_name in in_df.columns:
        raise ValueError(f'{var_name}  already exists in data')
    in_df[var_name] = np.nan
    update_hit_var(in_df, var_name, var_csvs, value_to_assign=value_to_assign, level=level)


def update_hit_var(in_df: pd.DataFrame, var_name: str, var_csvs: List[str], value_to_assign=1, level=None):
    """
    Updates variable in dataframe from variable csv by assigning value if taxa in variable csv.
    Will NOT overwrite values.
    :param in_df:
    :param var_name:
    :param var_csvs:
    :param value_to_assign:
    :param level:
    :return:
    """
    if level == 'genera' and var_name not in GENERA_VARS:
        raise ValueError(f'{var_name} not in GENERA_VARS')
    if level is None:
        if var_name in GENERA_VARS:
            level = 'genera'
        else:
            level = 'species'

    var_df = read_csv_list(var_csvs)

    # Assign value by matching accepted names
    in_df.loc[in_df[var_name].isna() & in_df[COL_NAMES['acc_name']].isin(
        var_df[COL_NAMES['acc_name']].dropna()), var_name] = value_to_assign
    if level != 'precise' or level == 'species':
        # Assign value at species level if value in nan.
        in_df.loc[in_df[var_name].isna() & in_df[COL_NAMES['acc_species']].isin(
            var_df[COL_NAMES['acc_name']].dropna()), var_name] = value_to_assign
        if COL_NAMES['acc_species'] in var_df:
            in_df.loc[in_df[var_name].isna() &
                      in_df[COL_NAMES['acc_species']].isin(
                          var_df[COL_NAMES['acc_species']].dropna()), var_name] = value_to_assign

    if level == 'genera':
        if 'Genus' not in var_df.columns:
            var_df['Genus'] = var_df[COL_NAMES['acc_name']].apply(get_genus_from_full_name)

        # Get values which haven't been set for the variable where the genus is in the variable dataset
        mask = in_df[var_name].isna() & in_df['Genus'].isin(var_df['Genus'].dropna())
        # mask_df = in_df[mask]
        in_df.loc[mask, var_name] = value_to_assign


def merge_new_vars_from_data(in_df: pd.DataFrame, var_names: List[str], var_csvs: List[str],
                             level=None) -> pd.DataFrame:
    var_df = read_csv_list(var_csvs)
    for var in var_names:
        if var not in var_df.columns:
            print(var_df.columns)
            raise ValueError(var)
    updated_df = merge_new_var_from_data(in_df, var_names[0], var_csvs, level=level)
    for var in var_names[1:]:
        updated_df = merge_new_var_from_data(updated_df, var, var_csvs, level=level)

    return updated_df


def merge_new_var_from_data(in_df: pd.DataFrame, var_name: str, var_csvs: List[str], level=None,
                            original_name_col: str = None) -> pd.DataFrame:
    if level is None:
        if var_name in GENERA_VARS + ["Habit"]:
            level = 'genera'
        else:
            level = 'species'
    var_df = read_csv_list(var_csvs)

    # Check for duplicates of accepted names and decide how to remove
    duplicates = var_df[var_df.duplicated([COL_NAMES['acc_name']], keep=False)]
    if len(duplicates.index) > 0:

        duplicates_to_keep = duplicates[duplicates[COL_NAMES['acc_name']] == duplicates[original_name_col]]

        if (len(duplicates_to_keep.index) == (len(duplicates.index) / 2)) and all(
                name in duplicates_to_keep[COL_NAMES['acc_name']].values for name in
                duplicates[COL_NAMES['acc_name']].values):

            var_df = var_df[(~var_df[COL_NAMES['acc_name']].isin(duplicates[COL_NAMES['acc_name']])) |
                            (var_df.index.isin(duplicates_to_keep.index))]

        else:
            print(duplicates[COL_NAMES['acc_name']])
            print(duplicates_to_keep)
            raise ValueError(f'variable with unresolved duplicates: {var_name}')

    # First match precisely
    var_accepted_names = var_df[[COL_NAMES['acc_name'], var_name]].dropna(subset=[COL_NAMES['acc_name'], var_name])
    updated_df = pd.merge(in_df, var_accepted_names, how='left', on=COL_NAMES['acc_name'])
    if level != 'precise':

        # Match by species
        species_values = pd.merge(updated_df.drop(columns=[var_name]), var_accepted_names, how='left',
                                  left_on=COL_NAMES['acc_species'], right_on=COL_NAMES['acc_name'])

        if len(species_values.index) > 0:
            updated_df[var_name].fillna(species_values[var_name], inplace=True)

        if COL_NAMES['acc_species'] in var_df:
            var_species_names = var_df[[COL_NAMES['acc_species'], var_name]].dropna(
                subset=[COL_NAMES['acc_species'], var_name])
            # Use first accepted species in list
            var_species_names.drop_duplicates(subset=[COL_NAMES['acc_species']], keep='first', inplace=True)
            # unmatched2 = updated_df[updated_df[var_name].isna()].drop(columns=[var_name])
            sp_values = pd.merge(updated_df.drop(columns=[var_name]), var_species_names, how='left',
                                 left_on=COL_NAMES['acc_species'], right_on=COL_NAMES['acc_species'])

            if len(sp_values) > 0:
                updated_df[var_name].fillna(sp_values[var_name], inplace=True)
    if level == 'genera':

        if 'Genus' not in var_df.columns:
            if original_name_col is None:
                var_df['Genus'] = var_df[COL_NAMES['acc_name']].apply(get_genus_from_full_name)
                genus_name_col = 'Genus'
            else:
                genus_name_col = original_name_col
        else:
            genus_name_col = 'Genus'
        var_genus_names = var_df[[genus_name_col, var_name]].dropna(subset=[genus_name_col, var_name])
        var_genus_names.drop_duplicates(subset=[genus_name_col], keep='first', inplace=True)
        genus_values = pd.merge(updated_df.drop(columns=[var_name]), var_genus_names, how='left',
                                left_on='Genus', right_on=genus_name_col)

        if len(genus_values.index) > 0:
            updated_df[var_name].fillna(genus_values[var_name], inplace=True)

    return updated_df


def modify_related_features(df: pd.DataFrame):
    df['Emergence'] = df[["Spines", "Hairs"]].max(axis=1)
    df['Medicinal'] = df[["Medicinal", "Antimalarial_Use", "History_Fever"]].max(axis=1)


def reset_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    # Following https://github.com/pandas-dev/pandas/issues/4094
    #     Update breaks dtypes which is a pain
    int_cols = DISCRETE_VARS

    for i_var in int_cols:
        if i_var in df.columns:
            df[i_var] = df[i_var].astype('float64')
        else:
            print(f'Warning: {i_var} specified in variables but not in Dataframe')

    float_cols = CONTINUOUS_VARS

    for f_var in float_cols:
        if f_var in df.columns:
            df[f_var] = df[f_var].astype('float64')
        else:
            print(f'Warning: {f_var} specified in variables but not in Dataframe')

    string_cols = NON_NUMERIC_TRAITS
    for s_var in string_cols:
        if s_var in df.columns:
            df[s_var] = df[s_var].astype('object')
        else:
            print(f'Warning: {s_var} specified in variables but not in Dataframe')

    return df


def attach_new_var_hits_absences(df: pd.DataFrame, var_name: str, hit_csvs: List[str], absence_csvs: List[str],
                                 level='precise',
                                 priority=1):
    if priority == 1:
        attach_new_var(df, var_name, hit_csvs, value_to_assign=1, level='precise')
        update_hit_var(df, var_name, absence_csvs, value_to_assign=0, level='precise')
    elif priority == 0:
        attach_new_var(df, var_name, absence_csvs, value_to_assign=0, level='precise')
        update_hit_var(df, var_name, hit_csvs, value_to_assign=1, level='precise')
    else:
        raise ValueError

    if level != 'precise' or level == 'species':
        if priority == 1:
            update_hit_var(df, var_name, hit_csvs, value_to_assign=1, level='species')
            update_hit_var(df, var_name, absence_csvs, value_to_assign=0, level='species')
        elif priority == 0:
            update_hit_var(df, var_name, absence_csvs, value_to_assign=0, level='species')
            update_hit_var(df, var_name, hit_csvs, value_to_assign=1, level='species')
        else:
            raise ValueError
    if level == 'genera':
        if priority == 1:
            update_hit_var(df, var_name, hit_csvs, value_to_assign=1, level='genera')
            update_hit_var(df, var_name, absence_csvs, value_to_assign=0, level='genera')

        elif priority == 0:
            update_hit_var(df, var_name, absence_csvs, value_to_assign=0, level='genera')
            update_hit_var(df, var_name, hit_csvs, value_to_assign=1, level='genera')
        else:
            raise ValueError


def main():
    trait_df = read_csv_list([encoded_traits_csv, logan_encoded_traits_csv])

    trait_df.drop(columns=['Source'], inplace=True)

    updated_trait_df = append_wcvp_taxa_to_trait_df(trait_df)
    updated_trait_df.reset_index(inplace=True, drop=True)

    # Resolve antimalarial activity to species
    duplicated_species = updated_trait_df[updated_trait_df['Accepted_Species'].duplicated(keep='first')][
        'Accepted_Species'].tolist()

    max_activities = dict()
    for sp in duplicated_species:
        max_act = updated_trait_df[updated_trait_df['Accepted_Species'] == sp][TARGET_COLUMN].max()
        max_activities[sp] = max_act

    for sp in max_activities.keys():
        updated_trait_df.loc[updated_trait_df['Accepted_Species'] == sp, TARGET_COLUMN] = max_activities[sp]
    # Drop infraspecies
    updated_trait_df = updated_trait_df[updated_trait_df['Accepted_Rank'] == 'Species']
    ### Precise
    attach_new_var(updated_trait_df, 'In_Malarial_Region',
                   [taxa_in_malarial_countries_csv, logan_taxa_in_malarial_countries_csv], level='precise')
    #### Species

    attach_new_var(updated_trait_df, 'Common_Name', [output_common_names_csv, output_logan_common_names_csv],
                   level='species')
    attach_new_var(updated_trait_df, 'Medicinal', [output_medicinal_csv, output_logan_medicinal_csv],
                   level='species')
    attach_new_var(updated_trait_df, 'Wiki_Page', [output_wiki_csv, output_logan_wiki_csv], level='species')
    updated_trait_df = merge_new_var_from_data(updated_trait_df, 'native_tdwg3_codes',
                                               [distributions_csv, logan_distributions_csv],
                                               level='species')
    update_hit_var(updated_trait_df, 'Antimalarial_Use', [output_malarial_csv, output_logan_malarial_csv],
                   level='species')
    updated_trait_df = merge_new_vars_from_data(updated_trait_df, ENVIRON_VARS,
                                                [compiled_climate_vars_csv, logan_compiled_climate_vars_csv],
                                                level='species')

    # Poison data covers all families
    attach_new_var_hits_absences(updated_trait_df, 'Poisonous', [output_poison_csv],
                                 [output_nonpoison_csv], level='species')

    update_hit_var(updated_trait_df, 'Alkaloids',
                   [rub_apoc_alkaloid_hits_output_csv, logan_alkaloid_hits_output_csv],
                   level='species')

    updated_trait_df['Tested_for_Alkaloids'] = (~updated_trait_df['Alkaloids'].isna()).astype(int)
    ### Genera
    updated_trait_df = merge_new_var_from_data(updated_trait_df, 'Habit', [habits_output_csv, logan_habits_output_csv],
                                               original_name_col='genus', level='genera')

    # First attach data by exact matches, then by species and then by genera (only nans are overwritten)
    # Spines
    attach_new_var_hits_absences(updated_trait_df, 'Spines', [spines_output_csv, logan_spines_output_csv],
                                 [no_spines_output_csv, logan_no_spines_output_csv], level='genera')

    attach_new_var(updated_trait_df, 'Hairs', [hairy_output_csv, logan_hairy_output_csv], level='genera')

    species_df = updated_trait_df[updated_trait_df['Accepted_Rank'] == 'Species']
    # Number of species in each genus (scaled by max)
    updated_trait_df['Richness'] = updated_trait_df.apply(
        lambda x: len(
            species_df[species_df['Genus'] == x.Genus]),
        axis=1)
    richest = updated_trait_df['Richness'].max()
    print(f'Richest genus: {updated_trait_df[updated_trait_df["Richness"] == 1]["Genus"].unique()}:{richest} species')
    # Scale richness:
    updated_trait_df['Richness'] = updated_trait_df['Richness'].divide(richest)

    print('Modifying features')
    modify_related_features(updated_trait_df)

    out_df = encode_habits(updated_trait_df)

    out_df = reset_dtypes(out_df)

    out_df.to_csv(TEMP_ALL_TRAITS_CSV)


if __name__ == '__main__':
    main()

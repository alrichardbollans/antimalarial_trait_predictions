import os

import pandas as pd
from apm_activity import cw_random_apm_data, cw_ethno_apm_data
from apm_activity.import_apm_data import _parsed_lit_apm_csv
from medicinal_usage import output_malarial_csv, output_medicinal_csv
from metabolites import all_taxa_metabolites_csv
from pkg_resources import resource_filename
from wcvp_download import wcvp_accepted_columns, plot_native_number_accepted_taxa_in_regions, get_distributions_for_accepted_taxa
from wcvp_name_matching import get_genus_from_full_name

from species_sampling import WCVP_VERSION, prediction_input_path

_output_path = resource_filename(__name__, 'outputs')
all_taxa_metabolite_data = pd.read_csv(all_taxa_metabolites_csv, index_col=0)

# ethno_data
malarial_use_data = pd.read_csv(output_malarial_csv, index_col=0)
medicinal_use_data = pd.read_csv(output_medicinal_csv, index_col=0)

# APM data
PLANT_TARGET_COLUMN = 'APM Activity'
_apm_data_from_literature = pd.read_csv(_parsed_lit_apm_csv,
                                        index_col=0)
_apm_data_from_literature['Genus'] = _apm_data_from_literature[wcvp_accepted_columns['name']].apply(
    get_genus_from_full_name)


def get_metab_info_for_species(sample_df: pd.DataFrame, outpath: str, filename: str) -> pd.DataFrame:
    metabolite_info = all_taxa_metabolite_data[
        all_taxa_metabolite_data[wcvp_accepted_columns['species_w_author']].isin(sample_df[wcvp_accepted_columns['species_w_author']].dropna().values)]
    metabolite_info = metabolite_info.sort_values(by=wcvp_accepted_columns['species_w_author']).reset_index(drop=True)

    metabolite_info.set_index(wcvp_accepted_columns['species_w_author']).to_csv(
        os.path.join(outpath, filename + 'metabolite_info.csv'))

    actives = metabolite_info[metabolite_info['active_chembl_compound'] == 1]
    actives.set_index(wcvp_accepted_columns['species_w_author']).to_csv(os.path.join(outpath, filename + 'active_metabolite_info.csv'))

    unknown_profiles = sample_df[
        ~sample_df[wcvp_accepted_columns['species_w_author']].isin(
            metabolite_info[wcvp_accepted_columns['species_w_author']].dropna().values)].reset_index(drop=True)
    unknown_profiles.set_index(wcvp_accepted_columns['species_w_author']).to_csv(
        os.path.join(outpath, filename + 'with_unknown_chemical_profiles.csv'))

    return metabolite_info


def get_apm_test_info_for_species(sample_df: pd.DataFrame, outpath: str, filename: str):
    # Previously tested species
    apm_info = _apm_data_from_literature[
        _apm_data_from_literature[wcvp_accepted_columns['species_w_author']].isin(sample_df[wcvp_accepted_columns['species_w_author']].dropna().values)]
    apm_info = apm_info.sort_values(by=wcvp_accepted_columns['species_w_author']).reset_index(drop=True)

    apm_info.set_index(wcvp_accepted_columns['species_w_author']).to_csv(
        os.path.join(outpath, filename + 'apm_info.csv'))

    #
    genera_with_no_activity_info = [c for c in sample_df['Genus'].dropna().unique() if c not in _apm_data_from_literature['Genus'].unique()]

    genera_with_no_activity_info_df = pd.DataFrame({'genera_with_no_activity_info': genera_with_no_activity_info})
    genera_with_no_activity_info_df.set_index('genera_with_no_activity_info').to_csv(
        os.path.join(outpath, filename + 'genera_with_no_activity_info.csv'))


def get_medicinal_use_info(sample_df: pd.DataFrame, outpath: str, filename: str):
    ethno_info = malarial_use_data[
        malarial_use_data[wcvp_accepted_columns['species_w_author']].isin(sample_df[wcvp_accepted_columns['species_w_author']].dropna().values)]
    ethno_info = ethno_info.sort_values(by=wcvp_accepted_columns['species_w_author']).reset_index(drop=True)
    ethno_info.set_index(wcvp_accepted_columns['species_w_author']).to_csv(
        os.path.join(outpath, filename + 'malarial_use_info.csv'))

    ethno_info = medicinal_use_data[
        medicinal_use_data[wcvp_accepted_columns['species_w_author']].isin(sample_df[wcvp_accepted_columns['species_w_author']].dropna().values)]
    ethno_info = ethno_info.sort_values(by=wcvp_accepted_columns['species_w_author']).reset_index(drop=True)
    ethno_info.set_index(wcvp_accepted_columns['species_w_author']).to_csv(
        os.path.join(outpath, filename + 'medicinal_use_info.csv'))


def get_data_summary(sample_df: pd.DataFrame, outpath: str, filename: str):
    sample_df.describe(include='all').to_csv(os.path.join(outpath, filename + 'summary.csv'))


def do_all_outputs_for_given_sample(sample_df: pd.DataFrame, outpath: str, filename: str):
    if not os.path.isdir(outpath):
        os.mkdir(outpath)
    sample_df.to_csv(os.path.join(outpath, filename + 'copy.csv'))
    get_data_summary(sample_df, outpath, filename)
    get_metab_info_for_species(sample_df, outpath, filename)
    get_apm_test_info_for_species(sample_df, outpath, filename)
    get_medicinal_use_info(sample_df, outpath, filename)
    ## Output distribution info
    dist_df = get_distributions_for_accepted_taxa(sample_df.drop_duplicates(subset=[wcvp_accepted_columns['species']], keep='first'),
                                                  wcvp_accepted_columns['species'],
                                                  include_extinct=True,
                                                  wcvp_version=WCVP_VERSION)
    dist_df.set_index(wcvp_accepted_columns['species_w_author']).to_csv(os.path.join(outpath, filename + 'species_distribution_data.csv'))
    plot_native_number_accepted_taxa_in_regions(sample_df, wcvp_accepted_columns['species'], outpath,
                                                filename + 'species_distributions.jpg', include_extinct=True,
                                                wcvp_version=WCVP_VERSION)


def do_all_outputs():
    # TODO: Compare with what Ed finds.
    random_sample = pd.read_csv(cw_random_apm_data, index_col=0)
    do_all_outputs_for_given_sample(random_sample, os.path.join(_output_path, 'random_sample_summary'), 'random_sample_')

    ethno_sample = pd.read_csv(cw_ethno_apm_data, index_col=0)
    do_all_outputs_for_given_sample(ethno_sample, os.path.join(_output_path, 'ethno_sample_summary'), 'ethno_sample_')

    ai_sample = pd.read_csv(os.path.join(prediction_input_path, 'sampled_from_predictions_V12.csv'), index_col=0)
    ai_sample['Genus'] = ai_sample[wcvp_accepted_columns['name']].apply(
        get_genus_from_full_name)
    do_all_outputs_for_given_sample(ai_sample, os.path.join(_output_path, 'ai_sample'), 'ai_sample_')

    all_sample = pd.concat([random_sample, ethno_sample, ai_sample])
    active_sample = all_sample[all_sample[PLANT_TARGET_COLUMN] == 1]
    do_all_outputs_for_given_sample(active_sample, os.path.join(_output_path, 'active_sample'), 'aactive_sample_')


def main():
    do_all_outputs()


if __name__ == '__main__':
    main()

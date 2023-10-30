import os

import pandas as pd
from apm_activity import compiled_extractions_apm_csv
from metabolites import all_taxa_metabolites_csv
from pkg_resources import resource_filename
from wcvp_download import wcvp_accepted_columns, plot_native_number_accepted_taxa_in_regions, get_distributions_for_accepted_taxa
from wcvp_name_matching import get_genus_from_full_name

from species_sampling import WCVP_VERSION

_output_path = resource_filename(__name__, 'outputs')
all_taxa_metabolite_data = pd.read_csv(all_taxa_metabolites_csv, index_col=0)

# APM data
PLANT_TARGET_COLUMN = 'APM Plant Activity'
all_compiled_apm_data = pd.read_csv(compiled_extractions_apm_csv,
                                    index_col=0)
all_compiled_apm_data['Genus'] = all_compiled_apm_data[wcvp_accepted_columns['name']].apply(
    get_genus_from_full_name)
extraction_data = all_compiled_apm_data.rename(columns={'APM Activity': PLANT_TARGET_COLUMN})


def get_metab_info_for_species(prediction_df: pd.DataFrame, metabolite_data: pd.DataFrame, outpath: str = None):
    metabolite_info = metabolite_data[
        metabolite_data[wcvp_accepted_columns['species_w_author']].isin(prediction_df[wcvp_accepted_columns['species_w_author']].values)]
    metabolite_info = metabolite_info.sort_values(by=wcvp_accepted_columns['species_w_author']).reset_index(drop=True)

    if outpath is not None:
        metabolite_info.set_index(wcvp_accepted_columns['species_w_author']).to_csv(
            os.path.join(outpath, 'metabolite_info_for_predicted_species.csv'))

        unknown_profiles = prediction_df[
            ~prediction_df[wcvp_accepted_columns['species_w_author']].isin(
                metabolite_info[wcvp_accepted_columns['species_w_author']].values)].reset_index(drop=True)
        unknown_profiles.set_index(wcvp_accepted_columns['species_w_author']).to_csv(
            os.path.join(outpath, 'predicted_species_with_unknown_chemical_profiles.csv'))

    return metabolite_info


def get_already_containing_apm_metabolites_info(prediction_df: pd.DataFrame, metabolite_data: pd.DataFrame, outpath: str):
    metab_info = get_metab_info_for_species(prediction_df, metabolite_data)
    actives = metab_info[metab_info['active_chembl_compound'] == 1]
    actives.set_index(wcvp_accepted_columns['species_w_author']).to_csv(os.path.join(outpath, 'active_compounds_in_predicted_species.csv'))


def get_already_tested_genus_info_for_species(prediction_df: pd.DataFrame, species_apm_data: pd.DataFrame, outpath: str):
    genera_with_no_activity_info = [c for c in prediction_df['Genus'].dropna().unique() if c not in species_apm_data['Genus'].unique()]

    genera_with_no_activity_info_df = pd.DataFrame({'genera_with_no_activity_info': genera_with_no_activity_info})
    genera_with_no_activity_info_df.set_index('genera_with_no_activity_info').to_csv(
        os.path.join(outpath, 'genera_in_predicted_species_with_no_activity_info.csv'))


def output_species_to_sample(prediction_df: pd.DataFrame, estimate_col: str, test_capacity: int, outpath: str = None):
    if test_capacity is None:
        test_capacity = len(prediction_df.index)
    _TAG = 'top_' + str(test_capacity)
    species_df = prediction_df[prediction_df[wcvp_accepted_columns['rank']] == 'Species'].sort_values(by=estimate_col, ascending=False).head(
        test_capacity)[
        [wcvp_accepted_columns['species_w_author'], wcvp_accepted_columns['species'], 'Genus', estimate_col, wcvp_accepted_columns['rank'],
         wcvp_accepted_columns['family']]]

    if outpath is not None:
        species_df.set_index(wcvp_accepted_columns['species_w_author']).to_csv(os.path.join(os.path.join(outpath, _TAG + '_species_to_sample.csv')))

        # Summarise
        species_df.describe(include='all').to_csv(os.path.join(outpath, _TAG + '_species_trait_summary.csv'))
    return species_df


def do_all_outputs(sampled_species: pd.DataFrame, outdir: str):
    ## Output metabolite info
    get_metab_info_for_species(sampled_species, all_taxa_metabolite_data, outdir)
    get_already_containing_apm_metabolites_info(sampled_species, all_taxa_metabolite_data, outdir)
    get_already_tested_genus_info_for_species(sampled_species, extraction_data, outdir)


    ## Output distribution info
    dist_df = get_distributions_for_accepted_taxa(sampled_species.drop_duplicates(subset=[wcvp_accepted_columns['species']], keep='first'),
                                                  wcvp_accepted_columns['species'],
                                                  include_extinct=True,
                                                  wcvp_version=WCVP_VERSION)
    dist_df.set_index(wcvp_accepted_columns['species_w_author']).to_csv(os.path.join(outdir, 'species_distribution_data.csv'))
    plot_native_number_accepted_taxa_in_regions(sampled_species, wcvp_accepted_columns['species'], outdir,
                                                'predicted_species_distributions.jpg', include_extinct=True,
                                                wcvp_version=WCVP_VERSION)


def main():

    # Look at species actually tested
    species_actually_sent = pd.read_csv(os.path.join('inputs', 'species_sent_for_sampling_with_SVC_trait_based_predictions_WCVP_V11.csv'))
    species_actually_sent['Genus'] = species_actually_sent[wcvp_accepted_columns['name']].apply(
        get_genus_from_full_name)
    do_all_outputs(species_actually_sent, _output_path)


if __name__ == '__main__':
    main()

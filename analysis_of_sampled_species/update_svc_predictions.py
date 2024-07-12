import os

import pandas as pd
import pandas.testing
from pkg_resources import resource_filename
from wcvp_download import wcvp_accepted_columns
from wcvp_name_matching import get_accepted_info_from_names_in_column

_predictions_output_dir = os.path.join('..', 'antimalarial_predictions', 'outputs', 'predictions')
_old_unlabelled_predictions_csv = os.path.join(_predictions_output_dir, 'unlabelled_output.csv')
_old_labelled_predictions_csv = os.path.join(_predictions_output_dir, 'labelled_output.csv')
_input_path = resource_filename(__name__, "inputs")
prediction_input_path = os.path.join(_input_path, 'prediction_samples')

updated_prediction_csv = os.path.join(prediction_input_path, 'SVC_trait_based_predictions_WCVP_V12.csv')
updated_unlabelled_prediction_csv = os.path.join(prediction_input_path, 'Unlabelled_SVC_trait_based_predictions_WCVP_V12.csv')
WCVP_VERSION = None


def update_predictions():
    # Previous data using V7
    _lab_input = pd.read_csv(_old_labelled_predictions_csv)
    _unlab_input = pd.read_csv(_old_unlabelled_predictions_csv)

    svc_predictions = pd.concat([_lab_input, _unlab_input])[
        ['Accepted_Name', 'Activity_Antimalarial', 'SVC Probability Estimate']]
    svc_predictions = svc_predictions.rename(columns={'Accepted_Name': 'previous_accepted_name'})

    svc_predictions_acc = get_accepted_info_from_names_in_column(svc_predictions, 'previous_accepted_name', wcvp_version=WCVP_VERSION)
    svc_predictions_acc = svc_predictions_acc[
        ['accepted_name_w_author', 'SVC Probability Estimate', 'Activity_Antimalarial',
         'accepted_ipni_id', 'accepted_name',
         'accepted_family', 'accepted_rank', 'accepted_species', 'accepted_species_w_author',
         'accepted_species_ipni_id', 'accepted_parent',
         'taxon_status', 'matched_by', 'previous_accepted_name', 'matched_name']]

    # Resolve duplicates
    svc_predictions_acc['equals_old_name'] = svc_predictions_acc['previous_accepted_name'] == svc_predictions_acc[
        'accepted_name']
    svc_predictions_acc['SVC Probability Estimate'] = svc_predictions_acc.groupby('accepted_species')[
        'SVC Probability Estimate'].transform('mean')

    svc_predictions_acc = svc_predictions_acc.drop_duplicates(subset=['accepted_species'])
    svc_predictions_acc = svc_predictions_acc.sort_values(by='SVC Probability Estimate', ascending=False).reset_index(drop=True)

    repeats = svc_predictions_acc[svc_predictions_acc['accepted_species'].duplicated(keep=False)]
    if len(repeats.index) > 0:
        repeats.to_csv('repeats.csv')
        raise ValueError

    svc_predictions_acc.to_csv(updated_prediction_csv)
    svc_predictions_acc_unlabelled = svc_predictions_acc[svc_predictions_acc['Activity_Antimalarial'].isna()]
    svc_predictions_acc_unlabelled.to_csv(updated_unlabelled_prediction_csv)


def update_sent_samples():
    # prediction samples. Needs updating when samples sent back.
    species_actually_sent = pd.read_csv(os.path.join(prediction_input_path, 'sampled_from_predictions.csv'))
    species_actually_sent = get_accepted_info_from_names_in_column(species_actually_sent, 'Sampled', wcvp_version=WCVP_VERSION)
    species_actually_sent.to_csv(os.path.join(prediction_input_path, 'sampled_from_predictions_V12.csv'))

    # ethno and random samples

def merge_sent_samples_with_svc_predictions():
    species_actually_sent = pd.read_csv(os.path.join('inputs', 'sampled_from_predictions_V12.csv'), index_col=0)

    with_info = pd.read_csv(updated_unlabelled_prediction_csv, index_col=0)[[wcvp_accepted_columns['species'], 'SVC Probability Estimate']]

    actually_sent_with_info = with_info[
        with_info[wcvp_accepted_columns['species']].isin(species_actually_sent[wcvp_accepted_columns['species']].values)]
    pandas.testing.assert_series_equal(actually_sent_with_info[wcvp_accepted_columns['species']],
                                       species_actually_sent[wcvp_accepted_columns['species']], check_index=False)
    actually_sent_with_info.to_csv(os.path.join(_input_path, 'species_sent_for_sampling_with_SVC_trait_based_predictions_WCVP_V12.csv'))


if __name__ == '__main__':
    update_predictions()
    update_sent_samples()

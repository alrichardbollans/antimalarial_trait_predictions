import os

import pandas as pd
from pkg_resources import resource_filename

from import_trait_data import IMPORTED_TRAIT_CSV, \
    TARGET_COLUMN, NUMERIC_TRAITS

vars_without_target_to_use = [x for x in NUMERIC_TRAITS if
                              (x not in [
                                  'Alkaloids'])] + [
                                 'Family', 'Genus', 'kg_mode']
vars_to_use_in_bias_analysis = vars_without_target_to_use + [TARGET_COLUMN]

ALL_TRAITS = pd.read_csv(IMPORTED_TRAIT_CSV, index_col='Accepted_Name')[
    vars_to_use_in_bias_analysis]
LABELLED_TRAITS = ALL_TRAITS[~(ALL_TRAITS[TARGET_COLUMN].isna())]
UNLABELLED_TRAITS = ALL_TRAITS[ALL_TRAITS[TARGET_COLUMN].isna()]

bias_output_dir = resource_filename(__name__, 'outputs')
WEIGHTED_LABELLED_DATA = os.path.join(bias_output_dir, 'weigthed_labelled_data_logit.csv')
WEIGHTED_UNLABELLED_DATA = os.path.join(bias_output_dir, 'weigthed_unlabelled_data_logit.csv')
# Choice motivated in discussion
apriori_known_biasing_features = ['Antimalarial_Use',
                                  'Tested_for_Alkaloids',
                                  'Medicinal',
                                  'In_Malarial_Region', 'Genus', 'Family']
apriori_features_to_target_encode = ['Genus', 'Family']
all_features_to_target_encode = apriori_features_to_target_encode + ['kg_mode']
# Write variables used
with open(os.path.join(bias_output_dir, 'variable_docs.txt'), 'w') as the_file:
    the_file.write(f'vars_to_use_in_bias_analysis:{vars_to_use_in_bias_analysis}\n')
    the_file.write(f'apriori_known_biasing_features:{apriori_known_biasing_features}\n')
    the_file.write(f'apriori_features_to_target_encode:{apriori_features_to_target_encode}\n')
    the_file.write(f'all_features_to_target_encode:{all_features_to_target_encode}\n')

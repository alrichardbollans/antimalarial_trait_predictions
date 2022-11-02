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

output_dir = resource_filename(__name__, 'outputs')

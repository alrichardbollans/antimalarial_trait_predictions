import pandas as pd
from pkg_resources import resource_filename

from import_trait_data import TAXA_IN_ALL_REGIONS_CSV, \
    TARGET_COLUMN, NUMERIC_TRAITS, ALK_CLASS_VARS

vars_without_target_to_use = [x for x in NUMERIC_TRAITS if
                              (x not in [
                                  'Alkaloids',
                                  'Steroids',
                                  'Cardenolides',
                                  'Hairs',
                                  'Spines',
                                  'AntiBac_Metabolites'] + ALK_CLASS_VARS)] + [
                                 'Family', 'Genus', 'kg_mode']
vars_to_use_in_bias_analysis = vars_without_target_to_use + [TARGET_COLUMN]

# LABELLED_TRAITS = pd.read_csv(LABELLED_TRAITS_CSV, index_col=0)[vars_to_use_in_bias_analysis]
# UNLABELLED_TRAITS = pd.read_csv(UNLABELLED_TRAITS_CSV, index_col=0)[vars_without_target_to_use]
# ALL_TRAITS = pd.read_csv(FINAL_TRAITS_CSV, index_col=0)[vars_to_use_in_bias_analysis]

ALL_TRAITS_IN_ALL_REGIONS = pd.read_csv(TAXA_IN_ALL_REGIONS_CSV, index_col='Accepted_Name')[
    vars_to_use_in_bias_analysis]
LABELLED_TRAITS_IN_ALL_REGIONS = ALL_TRAITS_IN_ALL_REGIONS[~(ALL_TRAITS_IN_ALL_REGIONS[TARGET_COLUMN].isna())]
UNLABELLED_TRAITS_IN_ALL_REGIONS = ALL_TRAITS_IN_ALL_REGIONS[ALL_TRAITS_IN_ALL_REGIONS[TARGET_COLUMN].isna()]

output_dir = resource_filename(__name__, 'outputs')

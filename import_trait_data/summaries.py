import os

import pandas as pd

from import_trait_data import IMPORTED_TRAIT_CSV, TARGET_COLUMN, IMPORT_OUTPUT_DIR

ALL_TRAITS = pd.read_csv(IMPORTED_TRAIT_CSV, index_col=0)
LABELLED_TRAITS = ALL_TRAITS[~(ALL_TRAITS[TARGET_COLUMN].isna())]
UNLABELLED_TRAITS = ALL_TRAITS[ALL_TRAITS[TARGET_COLUMN].isna()]

def summarise_traits():
    LABELLED_TRAITS.describe().to_csv(os.path.join(IMPORT_OUTPUT_DIR, 'labelled trait summary.csv'))
    UNLABELLED_TRAITS.describe().to_csv(os.path.join(IMPORT_OUTPUT_DIR, 'unlabelled trait summary.csv'))
    ALL_TRAITS.describe().to_csv(os.path.join(IMPORT_OUTPUT_DIR, 'all trait summary.csv'))

if __name__ == '__main__':
    summarise_traits()

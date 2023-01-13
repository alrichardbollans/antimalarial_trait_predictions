import os
from pkg_resources import resource_filename

from climate_vars import all_climate_names

NAME_COLUMNS = [
    "Accepted_Name",
    "Accepted_ID",
    "Family",
    "Genus",
    "Accepted_Rank",
    "Accepted_Species",
    "Accepted_Species_ID"
]

COLUMNS_TO_DROP = [
    "Ref_Alks",
    "Ref_H_Mal",
    "History_Fever",
    "Ref_H_Fever",
    "Ref_Activity",
    "Name",
    "Species"

]
ACCEPTED_NAME_COLUMN = "Accepted_Name"
TARGET_COLUMN = "Activity_Antimalarial"
IMPORT_INPUT_DIR = resource_filename(__name__, "inputs")
IMPORT_TEMP_OUTPUT_DIR = resource_filename(__name__, "temp_outputs")
TEMP_ALL_TRAITS_CSV = os.path.join(IMPORT_TEMP_OUTPUT_DIR, "all_traits.csv")

FAMILY_TAXA_CSV = os.path.join(IMPORT_TEMP_OUTPUT_DIR, 'wcvp_family.csv')
TRAIT_CSV_WITH_EXTRA_TAXA = os.path.join(IMPORT_TEMP_OUTPUT_DIR, "traits_with_extra_taxa.csv")

IMPORT_OUTPUT_DIR = resource_filename(__name__, "outputs")

IMPORTED_TRAIT_CSV = os.path.join(IMPORT_OUTPUT_DIR, "all_traits.csv")
IMPORTED_LABELLED_TRAIT_CSV = os.path.join(IMPORT_OUTPUT_DIR, "labelled_traits.csv")
IMPORTED_UNLABELLED_TRAIT_CSV = os.path.join(IMPORT_OUTPUT_DIR, "unlabelled_traits.csv")

FAMILIES_OF_INTEREST = [
    "Apocynaceae",
    "Rubiaceae",
    "Loganiaceae"
]

# Variables
TAXONOMIC_VARS = ['Family', 'Genus']
ENVIRON_VARS = all_climate_names
ENVIRON_VARS.remove('soil_soc')

SOIL_VARS = [c for c in ENVIRON_VARS if 'soil' in c] + ['soil_water_cap']

HABIT_COLS = ["herb",
              "liana",
              "succulent",
              "shrub",
              "subshrub",
              "tree"]

COMPOUND_PRESENCE_VARS = ["Alkaloids"]

GENERA_VARS = ["Spines",
               "Emergence",
               "Hairs"] + HABIT_COLS

MORPH_VARS = ["Spines",
              "Emergence",
              "Hairs"] + HABIT_COLS

BINARY_VARS = ["In_Malarial_Region", "Common_Name", "Poisonous", "Medicinal", "Wiki_Page",
               "Antimalarial_Use", "Emergence",
               'Tested_for_Alkaloids'] + HABIT_COLS + COMPOUND_PRESENCE_VARS

DISCRETE_VARS = BINARY_VARS
NON_NUMERIC_TRAITS = ['Genus', 'Family', 'kg_mode', 'kg_all', 'native_tdwg3_codes']
CONTINUOUS_VARS = [c for c in ENVIRON_VARS if
                   (c not in DISCRETE_VARS and c not in NON_NUMERIC_TRAITS)]

NUMERIC_TRAITS = DISCRETE_VARS + CONTINUOUS_VARS
TRAITS = NUMERIC_TRAITS + NON_NUMERIC_TRAITS
TRAITS_TO_DROP_AFTER_IMPORT = ["Spines", "Hairs"]
# Traits not to fill with 0s
TRAITS_WITH_NANS = HABIT_COLS + ENVIRON_VARS + COMPOUND_PRESENCE_VARS
# [
#     "Coloured_Latex",
#     "Left_Corolla"
# ]
TRAITS_WITHOUT_NANS = [x for x in TRAITS if x not in TRAITS_WITH_NANS]

with open(os.path.join(IMPORT_OUTPUT_DIR, 'variable_docs.txt'), 'w') as the_file:
    the_file.write(f'TAXONOMIC_VARS:{TAXONOMIC_VARS}\n')
    the_file.write(f'ENVIRON_VARS:{ENVIRON_VARS}\n')
    the_file.write(f'SOIL_VARS:{SOIL_VARS}\n')
    the_file.write(f'HABIT_COLS:{HABIT_COLS}\n')
    the_file.write(f'COMPOUND_PRESENCE_VARS:{COMPOUND_PRESENCE_VARS}\n')
    the_file.write(f'GENERA_VARS:{GENERA_VARS}\n')
    the_file.write(f'MORPH_VARS:{MORPH_VARS}\n')
    the_file.write(f'BINARY_VARS:{BINARY_VARS}\n')
    the_file.write(f'DISCRETE_VARS:{DISCRETE_VARS}\n')
    the_file.write(f'NON_NUMERIC_TRAITS:{NON_NUMERIC_TRAITS}\n')
    the_file.write(f'CONTINUOUS_VARS:{CONTINUOUS_VARS}\n')
    the_file.write(f'NUMERIC_TRAITS:{NUMERIC_TRAITS}\n')
    the_file.write(f'CONTINUOUS_VARS:{CONTINUOUS_VARS}\n')
    the_file.write(f'TRAITS:{TRAITS}\n')
    the_file.write(f'TRAITS_WITH_NANS:{TRAITS_WITH_NANS}\n')
    the_file.write(f'TRAITS_WITHOUT_NANS:{TRAITS_WITHOUT_NANS}\n')

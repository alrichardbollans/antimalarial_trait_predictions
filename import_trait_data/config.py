import os

from climate_vars import all_climate_names
from pkg_resources import resource_filename

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
    "Alkaloids_test_notes",
    'Alkaloid_classes',
    'Alkaloid_class_absences',
    'Extraction_Method',
    'Type_of_Test',
    'Cardenolides_details',
    'Cardenolides_Ref',
    'Steroids_details',
    'Steroids_Ref',
    'General_Phytochem_notes',
    "Alkaloid_mainclass(conal)",
    "Alkaloid_otherclasses",
    "Alkaloid_class_notes",
    "Ref_H_Mal",
    "History_Fever",
    "Ref_H_Fever",
    "Tested_Antimalarial",
    "Authors_Activity_Label",
    "Positive_Control_Used",
    "Given_Activities",
    "Ref_Activity",
    "General_notes",
    "MPNS_notes",
    "Details",
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
IMPORT_OUTPUT_DIR_FOR_MALARIAL_REGIONS = os.path.join(IMPORT_OUTPUT_DIR, 'in malarial regions')

FINAL_TRAITS_CSV = os.path.join(IMPORT_OUTPUT_DIR_FOR_MALARIAL_REGIONS, "all_traits.csv")
LABELLED_TRAITS_CSV = os.path.join(IMPORT_OUTPUT_DIR_FOR_MALARIAL_REGIONS, "labelled_final_trait.csv")
UNLABELLED_TRAITS_CSV = os.path.join(IMPORT_OUTPUT_DIR_FOR_MALARIAL_REGIONS, "unlabelled_final_trait.csv")
GENERA_LIST_CSV = os.path.join(IMPORT_OUTPUT_DIR_FOR_MALARIAL_REGIONS, "genera_list.csv")
SPECIES_LIST_CSV = os.path.join(IMPORT_OUTPUT_DIR_FOR_MALARIAL_REGIONS, "species_list.csv")

# Outputs not restricted by malarial regions
IMPORT_OUTPUT_DIR_NOT_RESTRICTED_BY_REGION = os.path.join(IMPORT_OUTPUT_DIR, 'not restricted by region')
TAXA_IN_ALL_REGIONS_CSV = os.path.join(IMPORT_OUTPUT_DIR_NOT_RESTRICTED_BY_REGION, "all_traits.csv")
TAXA_TESTED_FOR_ALK_CLASSES_IN_ALL_REGIONS_CSV = os.path.join(IMPORT_OUTPUT_DIR_NOT_RESTRICTED_BY_REGION,
                                                              "all_traits_tested_for_alk_classes.csv")
RAW_TRAITS_SUMMARY_CSV = os.path.join(IMPORT_OUTPUT_DIR_NOT_RESTRICTED_BY_REGION, "species_level_traits_summary.csv")
RAW_LABELLED_TRAITS_SUMMARY_CSV = os.path.join(IMPORT_OUTPUT_DIR_NOT_RESTRICTED_BY_REGION,
                                               "species_level_labelled_traits_summary.csv")
SPECIES_LEVEL_TRAITS = os.path.join(IMPORT_OUTPUT_DIR_NOT_RESTRICTED_BY_REGION,
                                    "species_level_traits.csv")
TEMP_ALKALOID_CLASS_DATA_CSV = os.path.join(IMPORT_TEMP_OUTPUT_DIR, 'alkaloid_classes.csv')

NON_MALARIAL_TRAITS_CSV = os.path.join(IMPORT_OUTPUT_DIR_NOT_RESTRICTED_BY_REGION, "non_malarial_traits.csv")
NON_MALARIAL_LABELLED_TRAITS_CSV = os.path.join(IMPORT_OUTPUT_DIR_NOT_RESTRICTED_BY_REGION, "labelled_non_malarial_traits.csv")
FAMILIES_OF_INTEREST = [
    "Apocynaceae",
    "Rubiaceae",
    "Loganiaceae"
]

# Variables
TAXONOMIC_VARS = ['Richness']
ENVIRON_VARS = all_climate_names

SOIL_VARS = [c for c in ENVIRON_VARS if 'soil' in c] + ['soil_water_cap']
# OHE Columns we will get. Note no examples with kg = 3
# KG_COLS = ['kg2_' + str(x + 1) for x in range(31) if x != 2]
# ENVIRON_VARS = INITIAL_CLIMATE_VARS + KG_COLS

HABIT_COLS = ["habit_hb",
              "habit_li",
              "habit_sc",
              "habit_sh",
              "habit_subsh",
              "habit_tr"]

ALK_CLASS_VARS = ['alk_diterpenoid', 'alk_imidazole', 'alk_indole', 'alk_indolizidine', 'alk_indolizine(to_confirm)',
                  'alk_isoquinoline', 'alk_mia', 'alk_misc. one n', 'alk_misc. two n', 'alk_monoterpene',
                  'alk_naphthyridine', 'alk_peptide', 'alk_piperidine', 'alk_purine', 'alk_pyrazine', 'alk_pyridine',
                  'alk_pyrrole', 'alk_pyrrolidine', 'alk_pyrrolizidine', 'alk_quinazoline', 'alk_quinoline',
                  'alk_quinolizidine', 'alk_simple amine', 'alk_spermidine', 'alk_steroidal', 'alk_to_confirm']

COMPOUND_PRESENCE_VARS = ["AntiBac_Metabolites", "Alkaloids", "Cardenolides",
                          "Steroids"] + ALK_CLASS_VARS

GENERA_VARS = ["Richness", "Spines",
               "Emergence",
               "Hairs"] + HABIT_COLS

MORPH_VARS = ["Spines",
              "Emergence",
              "Hairs"] + HABIT_COLS

BINARY_VARS = ["In_Malarial_Region", "Common_Name", "Poisonous", "Medicinal", "Wiki_Page",
               "Antimalarial_Use", "Spines", "Hairs", "Emergence",
               'Tested_for_Alkaloids'] + HABIT_COLS + COMPOUND_PRESENCE_VARS

DISCRETE_VARS = BINARY_VARS
NON_NUMERIC_TRAITS = ['Genus', 'Family', 'kg_mode', 'kg_all', 'native_tdwg3_codes']
CONTINUOUS_VARS = ['Richness'] + [c for c in ENVIRON_VARS if (c not in DISCRETE_VARS and c not in NON_NUMERIC_TRAITS)]

NUMERIC_TRAITS = DISCRETE_VARS + CONTINUOUS_VARS
TRAITS = NUMERIC_TRAITS + NON_NUMERIC_TRAITS

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

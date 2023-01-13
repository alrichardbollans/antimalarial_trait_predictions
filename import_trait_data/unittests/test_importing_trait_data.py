import ast
import os
import unittest

import numpy as np
import pandas as pd
from automatchnames import get_genus_from_full_name
from logan_metabolite_vars import logan_alkaloid_hits_output_csv
from metabolite_vars import rub_apoc_alkaloid_hits_output_csv
from pkg_resources import resource_filename

from import_trait_data import ACCEPTED_NAME_COLUMN, TRAITS_WITHOUT_NANS, TRAITS_WITH_NANS, \
    TARGET_COLUMN, GENERA_VARS, HABIT_COLS, DISCRETE_VARS, \
    IMPORTED_TRAIT_CSV, TEMP_ALL_TRAITS_CSV
from manually_collected_data import replace_yes_no_in_column

taxa_in_all_regions = pd.read_csv(IMPORTED_TRAIT_CSV)
labelled_output = taxa_in_all_regions[~(taxa_in_all_regions[TARGET_COLUMN].isna())]
unlabelled_output = taxa_in_all_regions[taxa_in_all_regions[TARGET_COLUMN].isna()]

_test_output_dir = resource_filename(__name__, 'test_outputs')


class Test(unittest.TestCase):

    def test_alk_species(self):
        rubapocalks = pd.read_csv(rub_apoc_alkaloid_hits_output_csv)
        log_alks = pd.read_csv(logan_alkaloid_hits_output_csv)

        all_alks = pd.concat([rubapocalks, log_alks])
        known_alk_species = all_alks['Accepted_Species'].unique().tolist()

        all_data_in_all_regions = pd.read_csv(IMPORTED_TRAIT_CSV)
        compiled_alk_species = all_data_in_all_regions[all_data_in_all_regions['Alkaloids'] == 1][
            'Accepted_Species'].unique().tolist()

        # In this case the species has been tested and found without alkaloids but a subspecies is found with alkaloids
        known_alk_species.remove('Oldenlandia corymbosa')
        self.assertCountEqual(known_alk_species, compiled_alk_species)

    def test_tested_alks(self):
        temp_df = pd.read_csv(TEMP_ALL_TRAITS_CSV, index_col=0)
        problem_df = temp_df[(temp_df['Tested_for_Alkaloids'] == 0) & (
                (temp_df['Alkaloids']==1) | (temp_df['Alkaloids']==0))]

        # Species which need updating in trait tables
        # Likely known alkaloids from KN but not been added to manual table
        self.assertEqual(len(problem_df.index), 0, msg=problem_df)

    def test_replace_yes_no_in_column(self):
        test_dict = {'0': ['no', 'No', ' no', 'No '],
                     '1': ['yes', 'Yes', 'yes ', ' Yes'],
                     'same': ['Not Detected', 'not detected', 'yest detect ed', 'not detected ']}
        test_df = pd.DataFrame(test_dict)

        replace_yes_no_in_column(test_df, '0')
        replace_yes_no_in_column(test_df, '1')
        replace_yes_no_in_column(test_df, 'same')
        for v in test_df['1'].values:
            self.assertEqual(v, 1)
        for v in test_df['0'].values:
            self.assertEqual(v, 0)

        self.assertListEqual(test_df['same'].values.tolist(), test_dict['same'])

    def test_output_types(self):

        # Test columns are almost the same
        self.assertEqual(labelled_output.columns.tolist(), unlabelled_output.columns.tolist())

        # Test Types
        float_columns = []
        str_columns = ['Family']
        int_columns = ['Poisonous', 'Medicinal', 'Antimalarial_Use',
                       'Alkaloids']

        self.assertEqual(labelled_output[TARGET_COLUMN].values.dtype, np.dtype('float64'))

        for c in float_columns:
            print(c)
            print(labelled_output[c].dtype)
            print(labelled_output[c].values.dtype)
            print(labelled_output[c].values)
            self.assertEqual(labelled_output[c].values.dtype, np.dtype(float))
            self.assertEqual(unlabelled_output[c].values.dtype, np.dtype(float))
            self.assertEqual(taxa_in_all_regions[c].values.dtype, np.dtype(float))

        for c in str_columns:
            self.assertEqual(labelled_output[c].values.dtype, np.dtype(object))
            self.assertEqual(unlabelled_output[c].values.dtype, np.dtype(object))
            self.assertEqual(taxa_in_all_regions[c].values.dtype, np.dtype(object))

        for c in int_columns:
            self.assertTrue((labelled_output[c].values.dtype == np.dtype('float64')) or (
                    labelled_output[c].values.dtype == np.dtype('int64')))
            self.assertTrue((unlabelled_output[c].values.dtype == np.dtype('float64')) or (
                    unlabelled_output[c].values.dtype == np.dtype('int64')))
            self.assertTrue((unlabelled_output[c].values.dtype == np.dtype('float64')) or (
                    taxa_in_all_regions[c].values.dtype == np.dtype('int64')))

    def test_genera_from_acc_names(self):
        self.assertEqual(get_genus_from_full_name('Danais'), 'Danais')
        self.assertEqual(get_genus_from_full_name('Danais spp.'), 'Danais')
        self.assertEqual(get_genus_from_full_name('Danais xanthorrhoea'), 'Danais')

    def test_genera_variables_are_all_the_same(self):

        for genus in taxa_in_all_regions['Genus'].unique():
            genus_df = taxa_in_all_regions[taxa_in_all_regions['Genus'] == genus]
            for var in GENERA_VARS:
                # These variables have some absence data specific to species
                if var not in ['Alkaloids', 'Spines'] and var in genus_df.columns:
                    print(genus)
                    print(var)
                    self.assertEqual(len(genus_df[var].unique().tolist()), 1)

    def test_habit_presence(self):
        nan_habits = labelled_output[labelled_output['succulent'].isna()]
        nan_habits.to_csv(os.path.join(_test_output_dir, 'unknown_labelled_habits.csv'))
        pd.DataFrame(nan_habits['Genus'].unique()).to_csv(
            os.path.join(_test_output_dir, 'unknown_labelled_genus_habits.csv'))
        self.assertEqual(len(nan_habits.index), 0)

        # This will fail as don't have for all data, important is the assertion above
        nan_habits = taxa_in_all_regions[taxa_in_all_regions['succulent'].isna()]
        nan_habits.to_csv(os.path.join(_test_output_dir, 'unknown_habits.csv'))
        pd.DataFrame(nan_habits['Genus'].unique()).to_csv(os.path.join(_test_output_dir, 'unknown_genus_habits.csv'))
        self.assertEqual(len(nan_habits.index), 0)

    def test_discrete_vars_are_discrete(self):
        for c in DISCRETE_VARS:
            if c in taxa_in_all_regions.columns:
                pd.testing.assert_series_equal(taxa_in_all_regions[c], taxa_in_all_regions[c].round())

    def test_kg_mode_in_all(self):
        modes = taxa_in_all_regions['kg_mode']
        all_kg = taxa_in_all_regions['kg_all']
        for idx, row in taxa_in_all_regions.iterrows():
            print(row['kg_mode'])
            print(row['kg_all'])
            if np.isnan(row['kg_mode']):
                self.assertTrue(np.isnan(row['kg_all']))

            else:
                self.assertIn(int(row['kg_mode']), ast.literal_eval(row['kg_all']))

    def test_climate_limits(self):
        bounds = {'kg_mode': [0, 30], 'bio1': [-20, 35],
                  'brkl_elevation': [-500, 6500], 'elevation': [-500, 6500],
                  'soil_ocs': [0, 212], 'soil_ph': [0, 92],
                  'latitude': [-90, 85], 'longitude': [-180, 180],
                  'soil_nitrogen': [0, 2500]}

        for var in bounds:
            print(var)
            min = taxa_in_all_regions[var].min()
            max = taxa_in_all_regions[var].max()

            self.assertGreaterEqual(min, bounds[var][0])
            self.assertLessEqual(max, bounds[var][1])

    def test_output_instances(self):
        ad_copy = taxa_in_all_regions.set_index(ACCEPTED_NAME_COLUMN, drop=False)
        taxa_in_all_regions_copy = taxa_in_all_regions.set_index(ACCEPTED_NAME_COLUMN, drop=False)

        def test_dict(sp, d):

            for trait in d:
                # OHEd variables
                if trait == 'habit':
                    if trait == 'habit':
                        cols = HABIT_COLS
                    # elif trait == 'kg_all':
                    #     cols = KG_COLS
                    #     tag = 'kg2_'
                    else:
                        raise ValueError
                    given_habits = d[trait].split('/')

                    for h in cols:
                        value = ad_copy.at[sp, h]
                        value2 = taxa_in_all_regions_copy.at[sp, h]
                        if any(gh == h for gh in given_habits):
                            self.assertEqual(1, value, msg=f'{sp}:{h}:{given_habits}')
                            self.assertEqual(1, value2, msg=f'{sp}:{h}:{given_habits}')
                        else:
                            self.assertTrue(np.isnan(value) or value == 0,
                                            msg=f'{sp}:{h}:{given_habits}:{value}')
                            self.assertTrue(np.isnan(value2) or value2 == 0,
                                            msg=f'{sp}:{h}:{given_habits}:{value2}')
                        if d[trait] == '':
                            self.assertTrue(np.isnan(value),
                                            msg=f'{sp}:{h}:{given_habits}:{value}')
                            self.assertTrue(np.isnan(value2),
                                            msg=f'{sp}:{h}:{given_habits}:{value2}')


                else:
                    try:
                        if np.isnan(d[trait]):
                            if trait in TRAITS_WITH_NANS + [TARGET_COLUMN]:
                                self.assertTrue(np.isnan(ad_copy.at[sp, trait]),
                                                msg=f'{sp}:{trait}:{ad_copy.at[sp, trait]}')
                                self.assertTrue(np.isnan(taxa_in_all_regions_copy.at[sp, trait]),
                                                msg=f'{sp}:{trait}:{taxa_in_all_regions_copy.at[sp, trait]}')
                            else:
                                try:
                                    self.assertEqual(0, ad_copy.at[sp, trait], msg=f'{sp}:{trait}')
                                    self.assertEqual(0, taxa_in_all_regions_copy.at[sp, trait], msg=f'{sp}:{trait}')
                                except KeyError as e:
                                    print(f'Warning:{e} not in data')
                        else:
                            try:
                                self.assertEqual(d[trait], ad_copy.at[sp, trait], msg=f'{sp}:{trait}')
                                self.assertEqual(d[trait], taxa_in_all_regions_copy.at[sp, trait], msg=f'{sp}:{trait}')
                            except KeyError as e:
                                print(f'Warning:{e} not in data')
                    except TypeError:
                        self.assertEqual(d[trait], ad_copy.at[sp, trait], msg=f'{sp}:{trait}')
                        self.assertEqual(d[trait], taxa_in_all_regions_copy.at[sp, trait], msg=f'{sp}:{trait}')

        aspi_dict = {TARGET_COLUMN: 1, 'Genus': 'Aspidosperma', 'Medicinal': 1, 'Poisonous': 1,
                     'Alkaloids': 1, 'Antimalarial_Use': 1, 'AntiBac_Metabolites': np.nan,
                     'Family': 'Apocynaceae', 'Spines': 0, 'Coloured_Latex': 1, 'Left_Corolla': 1, 'habit': 'tree',
                     'Steroids': np.nan, 'Cardenolides': np.nan,
                     'native_tdwg3_codes': "['BOL', 'BZE', 'BZL', 'BZC', 'CLM', 'GUY', 'PER', 'SUR', 'VEN', 'VNA']",
                     'alk_mia': 1, 'alk_indole': 1, 'alk_pyrrole': 0}
        test_dict('Aspidosperma parvifolium', aspi_dict)

        axill_dict = {'Genus': 'Baissea', 'Alkaloids': 0}
        test_dict('Baissea axillaris', axill_dict)
        pento_dict = {'Genus': 'Pentopetia', 'Alkaloids': 0}
        test_dict('Pentopetia albicans', pento_dict)

        vomica_dict = {TARGET_COLUMN: 0, 'Genus': 'Strychnos', 'Medicinal': 1, 'Poisonous': 1,
                       'Alkaloids': 1, 'Antimalarial_Use': 1, 'AntiBac_Metabolites': np.nan,
                       'Family': 'Loganiaceae', 'Spines': 1, 'habit': 'liana',
                       'Steroids': 1, 'Cardenolides': np.nan,
                       'native_tdwg3_codes': "['BAN', 'CBD', 'IND', 'LAO', 'MLY', 'MYA', 'SRL', 'THA', 'VIE']"}
        test_dict('Strychnos nux-vomica', vomica_dict)

        bert_dict = {TARGET_COLUMN: 0, 'Genus': 'Bertiera', 'Medicinal': 0, 'Poisonous': np.nan,
                     'Antimalarial_Use': 0,
                     'Family': 'Rubiaceae', 'Accepted_ID': '744422-1', 'Accepted_Rank': 'Species',
                     'Accepted_Species': 'Bertiera borbonica', 'Wiki_Page': 1, 'Alkaloids': np.nan, 'Spines': np.nan,
                     'Coloured_Latex': 0, 'Left_Corolla': np.nan,
                     'AntiBac_Metabolites': np.nan, 'habit': 'shrub/tree',
                     'native_tdwg3_codes': "['REU']"}
        test_dict('Bertiera borbonica', bert_dict)

        crypt_dict = {'habit': 'shrub/liana', 'native_tdwg3_codes': "['KEN', 'MOZ', 'TAN', 'ZAM', 'ZAI', 'ZIM']"}
        test_dict('Cryptolepis apiculata', crypt_dict)

        arac_dict = {'Genus': 'Arachnothryx', 'Alkaloids': 0, 'Medicinal': 0, 'Poisonous': np.nan,
                     'Antimalarial_Use': 0, 'Family': 'Rubiaceae', TARGET_COLUMN: np.nan,
                     'Spines': np.nan, 'Coloured_Latex': 0, 'Left_Corolla': np.nan,
                     'AntiBac_Metabolites': np.nan, 'habit': 'shrub/tree'}
        test_dict('Arachnothryx chiriquiana', arac_dict)

        cara_dict = {'Genus': 'Carapichea', 'Alkaloids': 1, 'Medicinal': 0, 'Poisonous': 0,
                     'Antimalarial_Use': 0, 'Family': 'Rubiaceae', TARGET_COLUMN: 1, 'Spines': np.nan,
                     'Coloured_Latex': 0, 'Left_Corolla': np.nan,
                     'AntiBac_Metabolites': np.nan, 'habit': 'shrub/tree/subshrub'}
        test_dict('Carapichea affinis', cara_dict)

        hima_dict = {'Genus': 'Himatanthus', 'Medicinal': 1, 'Poisonous': 0,
                     'Antimalarial_Use': 1,
                     'Family': 'Apocynaceae', 'Accepted_ID': '77136056-1', 'Accepted_Rank': 'Species',
                     'Accepted_Species': 'Himatanthus revolutus', 'Wiki_Page': 1, 'Spines': 0, 'Coloured_Latex': 0,
                     'Left_Corolla': 1, TARGET_COLUMN: np.nan, 'Alkaloids': np.nan, 'AntiBac_Metabolites': np.nan,
                     'habit': 'shrub', 'kg_all': "[1, 2, 3]"}
        test_dict('Himatanthus revolutus', hima_dict)

        mor_dict = {ACCEPTED_NAME_COLUMN: 'Morinda citrifolia', TARGET_COLUMN: 0, 'AntiBac_Metabolites': 1,
                    'Common_Name': 1}
        test_dict('Morinda citrifolia', mor_dict)

        asp_nit_dict = {ACCEPTED_NAME_COLUMN: 'Asperula nitida', TARGET_COLUMN: 1}
        test_dict('Asperula nitida', asp_nit_dict)

        gard_dict = {ACCEPTED_NAME_COLUMN: 'Gardenia ternifolia', TARGET_COLUMN: 1}
        test_dict('Gardenia ternifolia', gard_dict)

        prisma_dict = {ACCEPTED_NAME_COLUMN: 'Prismatomeris tetrandra', TARGET_COLUMN: 0}
        test_dict('Prismatomeris tetrandra', prisma_dict)

        perg_dict = {ACCEPTED_NAME_COLUMN: 'Pergularia daemia', TARGET_COLUMN: 0}
        test_dict('Pergularia daemia', perg_dict)

        mor_chry_dict = {ACCEPTED_NAME_COLUMN: 'Morinda chrysorhiza', TARGET_COLUMN: 0, 'AntiBac_Metabolites': np.nan,
                         'soil_nitrogen': 229.90625, 'soil_ph': 52.625}
        test_dict('Morinda chrysorhiza', mor_chry_dict)

        pent_dict = {ACCEPTED_NAME_COLUMN: 'Pentagonia gymnopoda',
                     'habit': 'shrub/tree', 'soil_ph': 53.4791679382324, 'kg_all': "[2]",
                     'kg_mode': 2}

        test_dict('Pentagonia gymnopoda', pent_dict)

        carismac_dict = {'Genus': 'Carissa', 'Accepted_Species': 'Carissa macrocarpa', 'Alkaloids': 0,
                         'habit': 'shrub'}
        test_dict('Carissa macrocarpa', carismac_dict)

        cariscar_dict = {'Genus': 'Carissa', 'Accepted_Species': 'Carissa carandas', 'Alkaloids': 1,
                         'habit': 'shrub'}
        test_dict('Carissa carandas', cariscar_dict)

    def test_related_features(self):
        def tests(df):
            # self.assertEqual(len(df[(df['Emergence'] == 0) & df['Spines'] == 1].index), 0)
            # self.assertEqual(len(df[(df['Emergence'] == 0) & df['Hairs'] == 1].index), 0)

            self.assertEqual(len(df[(df['Medicinal'] == 0) & df['Antimalarial_Use'] == 1].index), 0)
            if 'Steroids' in df.columns and 'Cardenolides' in df.columns:
                self.assertEqual(len(df[(df['Steroids'] == 0) & df['Cardenolides'] == 1].index), 0)

        tests(taxa_in_all_regions)
        tests(labelled_output)
        tests(unlabelled_output)

    def test_outputs(self):

        self.assertEqual(len(taxa_in_all_regions.index), len(labelled_output.index) + len(unlabelled_output.index))

        def test_for_duplication(df):
            duplicateRows = df[df.duplicated([ACCEPTED_NAME_COLUMN])]

            self.assertEqual(len(duplicateRows.index), 0, msg=duplicateRows.to_string())

        test_for_duplication(labelled_output)
        test_for_duplication(unlabelled_output)
        test_for_duplication(taxa_in_all_regions)

        def test_for_all_nans(df):
            mask = ' & '.join(
                [f'(df["{c}"].isna())' for c in
                 TRAITS_WITHOUT_NANS + TRAITS_WITH_NANS if c in df.columns])

            all_nan_values = df[eval(mask)]
            self.assertEqual(len(all_nan_values.index), 0,
                             msg=f'Samples with all nan values are present: {all_nan_values}')

        test_for_all_nans(taxa_in_all_regions)
        test_for_all_nans(labelled_output)

        test_for_all_nans(taxa_in_all_regions)

        test_for_duplication(unlabelled_output)

    def test_taxonomic_ranks(self):
        def do_rank_test(df):
            ranks = df['Accepted_Rank'].unique().tolist()
            self.assertEqual(ranks, ['Species'])

            self.assertEqual(len(df[df['Accepted_Species'].duplicated()]), 0)

        do_rank_test(labelled_output)
        do_rank_test(unlabelled_output)
        do_rank_test(taxa_in_all_regions)


if __name__ == '__main__':
    unittest.main()

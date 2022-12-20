import ast
import os

import pandas as pd
from getting_malarial_regions import get_tdwg3_codes, plot_countries
from pkg_resources import resource_filename

from bias_correction_and_summaries import quantbias_output_dir
from import_trait_data import IMPORTED_TRAIT_CSV, TARGET_COLUMN

_inputs_path = resource_filename(__name__, 'inputs')

_dist_output_dir = os.path.join(quantbias_output_dir, 'geography')


def OHE_dists(df: pd.DataFrame) -> pd.DataFrame:
    # OHE alks

    def reformat_dist_col(given_val):
        return ast.literal_eval(given_val)

    df['native_tdwg3_codes'] = df['native_tdwg3_codes'].apply(reformat_dist_col)

    multilabels = df['native_tdwg3_codes'].str.join('|').str.get_dummies()
    df = df.join(multilabels)

    return df


def plot_number_species_in_regions(df: pd.DataFrame, output_path: str, title: str = None):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as shpreader
    # Need fiona to read shapefiles
    import fiona
    tdwg3_shp = shpreader.Reader(
        os.path.join(_inputs_path, 'wgsrpd-master', 'level3', 'level3.shp'))
    tdwg3_region_codes = df['Region'].values
    min_val = df['Number of Native Species'].min()
    max_val = df['Number of Native Species'].max()
    norm = plt.Normalize(min_val, max_val)
    print('plotting countries')

    plt.figure(figsize=(40, 25))
    # plt.xlim(-210, 210)
    # plt.ylim(-60, 70)
    # plt.xticks(fontsize=30)
    # plt.yticks(fontsize=30)
    if title is not None:
        plt.title(title, fontsize=40)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linewidth=5)

    cmap = mpl.cm.get_cmap('coolwarm')
    for country in tdwg3_shp.records():

        tdwg_code = country.attributes['LEVEL3_COD']
        if tdwg_code in tdwg3_region_codes:

            # print(country.attributes['name_long'], next(earth_colors))
            ax.add_geometries([country.geometry], ccrs.PlateCarree(),
                              facecolor=cmap(
                                  norm(df.loc[df['Region'] == tdwg_code, 'Number of Native Species'].iloc[
                                           0])),
                              label=tdwg_code)

            x = country.geometry.centroid.x
            y = country.geometry.centroid.y

            # ax.text(x, y, tdwg_code, color='black', size=10, ha='center', va='center',
            #         transform=ccrs.PlateCarree())
        else:
            # print(f"code not in given malarial isocodes: {tdwg_code}")
            ax.add_geometries([country.geometry], ccrs.PlateCarree(),
                              facecolor='white',
                              label=tdwg_code)

    all_map_isos = [country.attributes['LEVEL3_COD'] for country in tdwg3_shp.records()]
    missed_names = [x for x in tdwg3_region_codes if x not in all_map_isos]
    print(f'iso codes not plotted on map: {missed_names}')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    plt.tight_layout()
    fig = plt.gcf()
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.175, 0.02, 0.65])
    cbar1 = fig.colorbar(sm, cax=cbar_ax)
    cbar1.ax.tick_params(labelsize=30)
    # cbar1 = plt.colorbar(sm, ax=ax, shrink=0.7)
    # cbar1.set_label('Test')
    # plt.tight_layout()
    # change the fontsize

    plt.savefig(output_path, dpi=50, bbox_inches='tight')


def get_region_dataframe(df: pd.DataFrame):
    ohe = OHE_dists(df)

    region_sums = {}
    for c in ohe.columns:
        if c not in df.columns:
            region_sums[c] = ohe[c].sum()

    dict_for_pandas = {'Region': region_sums.keys(), 'Number of Native Species': region_sums.values()}
    out_df = pd.DataFrame(dict_for_pandas)
    return out_df


def plot_active_data():
    all_traits = pd.read_csv(IMPORTED_TRAIT_CSV)
    all_active_traits = all_traits[all_traits[TARGET_COLUMN] == 1]
    all_active_traits = all_active_traits[all_active_traits['Accepted_Rank'] == 'Species']

    out_df = get_region_dataframe(all_active_traits)
    out_df.to_csv(
        os.path.join(_dist_output_dir, 'number_active_species_per_region.csv'),
        index=False)

    plot_number_species_in_regions(out_df,
                                   os.path.join(_dist_output_dir,
                                                'number_active_native_species_per_region.png'))
    return out_df


def get_labelled_region_df():
    all_traits = pd.read_csv(IMPORTED_TRAIT_CSV)
    all_labelled_traits = all_traits.dropna(subset=[TARGET_COLUMN])
    all_labelled_species = all_labelled_traits[all_labelled_traits['Accepted_Rank'] == 'Species']

    out_df = get_region_dataframe(all_labelled_species)
    out_df.to_csv(
        os.path.join(_dist_output_dir,
                     'number_labelled_species_per_region.csv'),
        index=False)

    return out_df


def plot_labelled_data():
    out_df = get_labelled_region_df()
    plot_number_species_in_regions(out_df,
                                   os.path.join(_dist_output_dir,
                                                'number_labelled_native_species_per_region.png'))


def get_all_region_df():
    all_traits = pd.read_csv(IMPORTED_TRAIT_CSV)

    all_species = all_traits[all_traits['Accepted_Rank'] == 'Species']

    out_df = get_region_dataframe(all_species)
    out_df.to_csv(
        os.path.join(_dist_output_dir,
                     'number_species_per_region.csv'),
        index=False)
    return out_df


def plot_all_data():
    out_df = get_all_region_df()

    plot_number_species_in_regions(out_df,
                                   os.path.join(_dist_output_dir,
                                                'number_native_species_per_region.png'))


def plot_proportion_active():
    """ This is a little misleading as countries with only one species stand out
    but it does highlight some particularly active countries e.g. Burkina, Borneo, Thailand etc.
    """

    all_traits = pd.read_csv(IMPORTED_TRAIT_CSV)
    all_labelled_traits = all_traits.dropna(subset=[TARGET_COLUMN])
    all_labelled_species = all_labelled_traits[all_labelled_traits['Accepted_Rank'] == 'Species']
    num_labelled_df = get_region_dataframe(all_labelled_species)

    all_traits = pd.read_csv(IMPORTED_TRAIT_CSV)
    all_active_traits = all_traits[all_traits[TARGET_COLUMN] == 1]
    all_active_traits = all_active_traits[all_active_traits['Accepted_Rank'] == 'Species']

    num_active_df = get_region_dataframe(all_active_traits)

    num_labelled_df.rename(columns={'Number of Native Species': 'num_labelled'}, inplace=True)
    num_active_df.rename(columns={'Number of Native Species': 'num_active'}, inplace=True)
    proportion_df = pd.merge(num_active_df, num_labelled_df)

    proportion_df['Number of Native Species'] = proportion_df['num_active'] / proportion_df['num_labelled']
    proportion_df.to_csv(os.path.join(_dist_output_dir,
                                      'proportion_active_species_per_region.csv'))
    plot_number_species_in_regions(proportion_df,
                                   os.path.join(_dist_output_dir,
                                                'proportion_active_native_species_per_region.png'))

    prop_df_with_at_least_two_labelled = proportion_df[proportion_df['num_labelled'] > 2]

    plot_number_species_in_regions(prop_df_with_at_least_two_labelled,
                                   os.path.join(_dist_output_dir,
                                                'proportion_active_gt2lab_species_per_region.png'))


def plot_malarial_regions():
    codes = get_tdwg3_codes()
    plot_countries(codes, 'Historical Malarial Regions',
                   os.path.join(_dist_output_dir, 'malarial_countries.png'))


def main():
    plot_active_data()
    plot_labelled_data()
    plot_proportion_active()
    plot_all_data()
    plot_malarial_regions()


if __name__ == '__main__':
    main()

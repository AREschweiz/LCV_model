###################
# This script generates a csv file for the estimation of the discrete choice models of the module End Tour.
###################

import numpy as np
import pandas as pd
import openmatrix as omx
from typing import Any, Dict, List
from pathlib import Path

folder_project = Path.cwd().parent


#%%

use_new_skim = True

path_zone_stats = folder_project / 'data' / 'zone_stats_2013_ZIP.csv'

# Official LCV cost figures, corrected by producer price index (2013, 2016)
chf_per_hour: float = 48.90 * (100.3 / 95.7)
chf_per_km: float = 0.5553 * (100.3 / 95.7)

n_zones_npa: int = 3187

bin_sizes_init: List[int] = [200, 600, n_zones_npa - 200 - 600] # for first trips of a tour, all destinations are allowed.
bin_sizes_interm = bin_sizes_init.copy()
bin_sizes_interm[-1] = bin_sizes_interm[-1] - 1 # in case trip_id>0, the base is not an allowed destination.

n_samples: List[int] = [100, 100, 100]



num_alt: int = sum(n_samples)

seed: int = 739150
# seed: int = 193593
# seed: int = 381055

n_holdout_sets = 0
holdout_set = 1


#%%

print('Reading data...')

#    with omx.open_file(folder_project + 'Sources/from_ARE/new matrices/tt_matrix_PLZ.omx', 'r') as omx_file:
with omx.open_file(str(folder_project / 'data' / 'PLZ_TTC_BelastungenNPVM2017.omx'), 'r') as omx_file:
    tt_matrix = np.array(omx_file[omx_file.list_matrices()[0]]).astype(np.float32)
    zone_mapping = omx_file.mapping('NO')

with omx.open_file(str(folder_project / 'data' / 'PLZ_DIS_BelastungenNPVM2017.omx'), 'r') as omx_file:
    dist_matrix = np.array(omx_file[omx_file.list_matrices()[0]]).astype(np.float32)

for i in range(tt_matrix.shape[0]):
    tt_matrix[i, i] = 0.5*tt_matrix[i, np.argsort(tt_matrix[i, :])[1]]
    dist_matrix[i, i] = 0.5*dist_matrix[i, np.argsort(dist_matrix[i, :])[1]]

cost_matrix: np.ndarray = (chf_per_hour / 60) * tt_matrix + chf_per_km * dist_matrix

zone_mapping_inv: Dict[int, int] = dict((v, k) for k, v in zone_mapping.items())


trips_df = pd.read_csv(folder_project / 'data' / 'trips_lcv_data.csv', sep=',')

# remove internal trips where the reported distance is more than 2 times larger than the distance to the closest PLZ neighbor
# the factor 2 is here because we assumed the internal distance to be half of the distance to the nearest neighbor
trips_df = trips_df[~((trips_df['ORIG'] == trips_df['DEST']) & (2 * 2*trips_df['DIST'] <= trips_df['DIST_SURVEY']))]

cum_bin_sizes_init: List[int] = [0] + [int(value) for value in np.cumsum(bin_sizes_init)]
cum_bin_sizes_interm: List[int] = [0] + [int(value) for value in np.cumsum(bin_sizes_interm)]
cum_n_samples: List[int] = [0] + [int(value) for value in np.cumsum(n_samples)]

zone_stats = pd.read_csv(path_zone_stats, sep=';')
population: np.ndarray = zone_stats['Pop'].values
jobs: np.ndarray = zone_stats['Jobs'].values
land_use: np.ndarray = zone_stats['LandUse'].values
area: np.ndarray = zone_stats['Area'].values
population[pd.isna(population)] = 0.0

LowDensity = (population/area <= 100) & (jobs/area<= 100)
Residential = (population/area > 100) & (2*jobs <= population)
Intermediary = (jobs/area <= 3000) & np.logical_not(LowDensity) & np.logical_not(Residential)
EmploymentNode = np.logical_not(Intermediary) & np.logical_not(LowDensity) & np.logical_not(Residential)
#%%

print('Constructing choice data...')

estimation_data: List[List[Any]] = []

np.random.seed(seed)

for i, row in enumerate(trips_df.to_dict('records')):
    zone_orig: int = zone_mapping.get(row['ORIG'])
    zone_dest: int = zone_mapping.get(row['DEST'])

    oid: int = row['OID']
    tour_id: int = row['TOUR_ID']
    trip_id: int = row['TRIP_ID']

    purpose_goods: int = row['PURPOSE_GOODS'] if not pd.isna(row['PURPOSE_GOODS']) else 0
    purpose_service: int = row['PURPOSE_SERVICE'] if not pd.isna(row['PURPOSE_SERVICE']) else 0

    curb_weight: int = row['CURB_WEIGHT']
    branch: int = row['BRANCH']
    statistical_weight: float = row['STATISTICAL_WEIGHT'] / 10000

    if trip_id == 0:
        zone_base: int = zone_orig

    if zone_orig is None or zone_dest is None:
        continue

    if (trip_id>0 and (zone_dest == zone_base)):
        # new: condition trip_id>0
        # this is necessary because in the estimation we have this option.
        continue

    dest_zones_sorted: np.ndarray = np.argsort(cost_matrix[zone_orig, :])

    if trip_id > 0: # in this case, the base is not an allowed destination (it would be considered a return trip)
        dest_zones_sorted = dest_zones_sorted[dest_zones_sorted != zone_base]
        bin_sizes = bin_sizes_interm.copy()
        cum_bin_sizes = cum_bin_sizes_interm.copy()
    else:
        bin_sizes = bin_sizes_init.copy()
        cum_bin_sizes = cum_bin_sizes_init.copy()

    chosen_alt: int = np.where(dest_zones_sorted == zone_dest)[0][0]

    sampled_alts: List[int] = []

    for bin_id, bin_size in enumerate(bin_sizes):
        bin_lower = cum_bin_sizes[bin_id]
        bin_upper = cum_bin_sizes[bin_id + 1]
        bin_n_samples = n_samples[bin_id]

        bin_options = dest_zones_sorted[bin_lower:bin_upper]

        # add chosen destination as alternative.
        if bin_lower <= chosen_alt < bin_upper:
            sampled_alts.append(zone_dest)
            bin_options = [option for option in bin_options if option != zone_dest]
            bin_n_samples -= 1

        # sample randomly selected alternatives
        rng = np.random.default_rng()
        for option in rng.choice(bin_options, size=bin_n_samples, replace=False):
            sampled_alts.append(option)

    is_first_leg: bool = (trip_id == 0)

    tmp_new_record: List[Any] = [
        oid, tour_id, statistical_weight,
        is_first_leg, purpose_goods, purpose_service, curb_weight, branch]
    for j, zone_alt in enumerate(sampled_alts):

        for bin_id in range(len(n_samples)):
            if cum_n_samples[bin_id] <= j < cum_n_samples[bin_id + 1]:
                p = bin_sizes[bin_id] / n_samples[bin_id]

        tmp_new_record = tmp_new_record + [
            int(zone_alt == zone_dest), cost_matrix[zone_orig, zone_alt],
            np.max([0, cost_matrix[zone_orig, zone_alt]-50]),
            population[zone_alt], jobs[zone_alt], LowDensity[zone_alt], Residential[zone_alt],
            Intermediary[zone_alt], EmploymentNode[zone_alt], int(zone_alt == zone_orig), p]

    estimation_data.append(tmp_new_record.copy())

df_columns: List[str] = [
    'OID', 'TOUR_ID', 'STATISTICAL_WEIGHT',
    'IS_FIRST_LEG', 'PURPOSE_GOODS', 'PURPOSE_SERVICE', 'CURB_WEIGHT', 'BRANCH']
for alt in range(sum(n_samples)):
    df_columns = df_columns + [
        f'CHOICE_{alt + 1}',
        f'COST_{alt + 1}',
        f'COST_50_{alt + 1}',
        f'POP_{alt + 1}', f'JOBS_{alt + 1}', f'LowDensity_{alt + 1}',
        f'Residential_{alt + 1}', f'Intermediary_{alt + 1}', f'EmploymentNode_{alt + 1}',
        f'isInternal_{alt + 1}', f'P_{alt + 1}']

estimation_data_df = pd.DataFrame(np.array(estimation_data), columns=df_columns)

for i in estimation_data_df.index:
    estimation_data_df.loc[i, 'CHOICE'] = 1 + np.where(estimation_data_df.loc[i, [f"CHOICE_{alt + 1}" for alt in range(num_alt)]] == 1)[0][0]


# Only keep the new CHOICE column, drop the individual ones
estimation_data_df.drop(columns=[f"CHOICE_{alt + 1}" for alt in range(num_alt)], inplace=True)

'''
if n_holdout_sets > 0:
    print(f'Removing holdout set {holdout_set}...')
    holdout_rows = np.arange((holdout_set - 1), len(estimation_data_df), n_holdout_sets)
    estimation_data_final_df = estimation_data_df.drop(index=holdout_rows)
else:
    estimation_data_final_df = estimation_data_df.copy()
'''
print('Exporting choice data...')
estimation_data_df.to_csv(folder_project / 'data' / 'estimation_data_for_next_stop_location.csv', sep=';', index=False)
#%%






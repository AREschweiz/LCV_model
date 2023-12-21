###################
# This script generates a csv file for the estimation of the discrete choice models of the module End Tour.
###################

import numpy as np
import pandas as pd
import openmatrix as omx
from typing import Any, Dict, List
from pathlib import Path

folder_project = Path.cwd().parent

path_zone_stats = folder_project / 'data' / 'zone_stats_2013_ZIP.csv'

# Official LCV cost figures, corrected by producer price index (2013, 2016)
#chf_per_hour: float = 48.90 * (100.3 / 95.7)
#chf_per_km: float = 0.5553 * (100.3 / 95.7)
chf_per_hour: float = 48.7533 * (100.3 / 95.7)
chf_per_km: float = 0.5536 * (100.3 / 95.7)

seed: int = 739150
# seed: int = 193593

#%%

print('Reading data...')

# Skim matrices
with omx.open_file(str(folder_project / 'data' / 'PLZ_TTC_BelastungenNPVM2017.omx'), 'r') as omx_file:
    tt_matrix = np.array(omx_file[omx_file.list_matrices()[0]]).astype(np.float32)
    zone_mapping = omx_file.mapping('NO')

with omx.open_file(str(folder_project / 'data' / 'PLZ_DIS_BelastungenNPVM2017.omx'), 'r') as omx_file:
    dist_matrix = np.array(omx_file[omx_file.list_matrices()[0]]).astype(np.float32)

for i in range(tt_matrix.shape[0]):
    tt_matrix[i, i] = 0.5*tt_matrix[i, np.argsort(tt_matrix[i, :])[1]]
    dist_matrix[i, i] = 0.5*dist_matrix[i, np.argsort(dist_matrix[i, :])[1]]

# Generalized travel cost
cost_matrix: np.ndarray = (chf_per_hour / 60) * tt_matrix + chf_per_km * dist_matrix

zone_mapping_inv: Dict[int, int] = dict((v, k) for k, v in zone_mapping.items())

# Trips
trips_df = pd.read_csv(folder_project / 'data' / 'trips_lcv_data.csv', sep=',')

# Explanatory variables
zone_stats = pd.read_csv(path_zone_stats, sep=';')
population: np.ndarray = zone_stats['Pop'].values
jobs: np.ndarray = zone_stats['Jobs'].values
land_use: np.ndarray = zone_stats['LandUse'].values

population[pd.isna(population)] = 0.0

#%%

print('Compute accessibility per zone')
accessibility = np.matmul(np.exp(-0.05 * tt_matrix), population + jobs)
print(f'Average accessibility : {np.mean(accessibility)}')

#%%

print('Constructing choice data...')

estimation_data: List[List[Any]] = []

np.random.seed(seed)

for i, row in enumerate(trips_df.to_dict('records')):
    zone_orig: int = zone_mapping.get(row['ORIG'])
    zone_dest: int = zone_mapping.get(row['DEST'])

    oid: int = row['OID']
    trip_id: int = row['TRIP_ID']
    n_trips: int = row['N_TRIPS']

    if trip_id == 0:
        zone_base: int = zone_orig
        continue

    if zone_orig is None or zone_base is None:
        continue

    purpose_goods: int = row['PURPOSE_GOODS'] if not pd.isna(row['PURPOSE_GOODS']) else 0
    purpose_service: int = row['PURPOSE_SERVICE'] if not pd.isna(row['PURPOSE_SERVICE']) else 0

    curb_weight: int = row['CURB_WEIGHT']
    branch: int = row['BRANCH']
    statistical_weight: float = row['STATISTICAL_WEIGHT'] / 10000

    is_return: bool = ((zone_dest == zone_base) or (trip_id == (n_trips - 1)))

    tmp_new_record: List[Any] = [
        oid, statistical_weight, is_return,
        purpose_goods, purpose_service, curb_weight, branch,
        trip_id, accessibility[zone_orig],
        cost_matrix[zone_orig, zone_base]]

    estimation_data.append(tmp_new_record.copy())

df_columns: List[str] = [
    'OID', 'STATISTICAL_WEIGHT',
    'IS_RETURN', 'PURPOSE_GOODS', 'PURPOSE_SERVICE', 'CURB_WEIGHT', 'BRANCH',
    'TRIP_ID', 'ACCESSIBILITY',
    'COST_RETURN']

estimation_data_df = pd.DataFrame(np.array(estimation_data), columns=df_columns)

#%%

print('Exporting choice data...')

estimation_data_df.to_csv(
    folder_project / 'data' / 'estimation_data_for_end_tour.csv', sep=';', index=False)

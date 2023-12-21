###################
# This script processes the survey data to create datasets with tours, shipments and trips.
###################

import numpy as np
import pandas as pd
import shapefile as shp
import openmatrix as omx
from pathlib import Path
from typing import Any, Dict, List

folder_project = Path.cwd().parent


#%%

# Same function as in support.py
def write_shape_line(
    path: str,
    data: pd.DataFrame,
    columns_coords: List[str],
    decimal_coords: int = 2,
    decimal_cols: int = 2
) -> None:
    """
    Writes a DataFrame with coordinates to shapefile.
    """
    n_records = data.shape[0]

    if n_records == 0:
        return

    data.index = np.arange(n_records)

    for col_name in columns_coords:
        if col_name not in data.columns:
            raise Exception(f"Column '{col_name}' not found in data.")

    orig_x = np.array(np.round(data[columns_coords[0]], decimal_coords), dtype=float)
    orig_y = np.array(np.round(data[columns_coords[1]], decimal_coords), dtype=float)
    dest_x = np.array(np.round(data[columns_coords[2]], decimal_coords), dtype=float)
    dest_y = np.array(np.round(data[columns_coords[3]], decimal_coords), dtype=float)

    w = shp.Writer(path)
    for col in range(len(data.columns)):
        col_name = data.columns[col]

        if type(data.at[0, col_name]) == str:
            w.field(col_name, 'C')

        elif col_name not in columns_coords:
            tmp_decimals = 0 if type(data.at[0, col_name]) in [int, np.int32, np.int64] else decimal_cols
            max_col_length = len(str(data[col_name].abs().max())) + tmp_decimals
            w.field(col_name, 'N', size=max_col_length, decimal=tmp_decimals)

    dbf_data = np.array(data.drop(columns=columns_coords), dtype=object)

    for i in range(n_records):
        w.line([[[orig_x[i], orig_y[i]], [dest_x[i], dest_y[i]]]])
        w.record(*dbf_data[i, :])

        if i % 100 == 0:
            print('\t' + str(round((i / n_records) * 100, 1)) + '%', end='\r')

    w.close()

    print('\t100%', end='\r')


#%% Read skims
with omx.open_file(str(folder_project / 'data' / 'PLZ_TTC_BelastungenNPVM2017.omx'), 'r') as omx_file:
    tt_matrix = np.array(omx_file[omx_file.list_matrices()[0]]).astype(np.float32)
    zone_mapping = omx_file.mapping('NO')

with omx.open_file(str(folder_project / 'data' / 'PLZ_DIS_BelastungenNPVM2017.omx'), 'r') as omx_file:
    dist_matrix = np.array(omx_file[omx_file.list_matrices()[0]]).astype(np.float32)

# Fill in intrazonals
for i in range(tt_matrix.shape[0]):
    tt_matrix[i, i] = 0.5*tt_matrix[i, np.argsort(tt_matrix[i, :])[1]]
    dist_matrix[i, i] = 0.5*dist_matrix[i, np.argsort(dist_matrix[i, :])[1]]

# Centroid coordinates, created in QGIS based on PLZO_PLZ.shp
centroids = dict(
    (row['PLZ'], (row['X'], row['Y']))
    for row in pd.read_csv(
        folder_project / 'data' / 'PLZO_PLZ_centroids.csv', sep=',').to_dict('records'))

print(len(centroids), 'centroids')


#%% Read LCV data

sheet = pd.read_excel(folder_project / "data" / "LWE_2013_TRANSPORT_scripts_only-signed.xlsm")


#%% Define shipments

shipments_df = sheet[[
    'OID', 'Quelle_NPA', 'Ziel_NPA', 'MAIN_USE_GOODS_TRANSPORT', 'MAIN_USE_SERVICE',
    'CURB_WEIGHT', 'NOGA_2008',
    'quelle_km', 'ziel_km', 'Anzahl_Stopps', 'wh_tot_cal']]
shipments_df = shipments_df.sort_values(['OID', 'quelle_km'])

# Remove users who made simplified tours (those with many stops)
oid_with_simplified_tours = np.unique(shipments_df.loc[~pd.isna(shipments_df['Anzahl_Stopps']), 'OID'])
shipments_df = shipments_df[~shipments_df['OID'].isin(oid_with_simplified_tours)]
shipments_df = shipments_df.drop(columns='Anzahl_Stopps')

n_shipments = len(shipments_df)
shipments_df.index = [i for i in range(n_shipments)]

print(n_shipments, 'shipments')

# Gives the purpose given the OID
oid_purpose_goods: Dict[int, int] = dict((row['OID'], row['MAIN_USE_GOODS_TRANSPORT']) for row in shipments_df.to_dict('records'))
oid_purpose_service: Dict[int, int] = dict((row['OID'], row['MAIN_USE_SERVICE']) for row in shipments_df.to_dict('records'))

# Gives the curb weight, branch, and statistical weight given the OID
oid_curb_weight: Dict[int, int] = dict((row['OID'], row['CURB_WEIGHT']) for row in shipments_df.to_dict('records'))
oid_branch: Dict[int, int] = dict((row['OID'], row['NOGA_2008']) for row in shipments_df.to_dict('records'))
oid_statistical_weight: Dict[int, float] = dict((row['OID'], row['wh_tot_cal']) for row in shipments_df.to_dict('records'))


#%% Define stops and legs

# keep='first' ensures that we only keep the NPA of the origin with the smallest mileage -> this will be the base.
oids = shipments_df[[
    'OID', 'MAIN_USE_GOODS_TRANSPORT', 'MAIN_USE_SERVICE',
    'CURB_WEIGHT', 'NOGA_2008', 'wh_tot_cal', 'Quelle_NPA']].copy()
oids = oids.drop_duplicates(subset=['OID'], ignore_index=True, keep='first')

oids = oids.rename(columns={
    'MAIN_USE_GOODS_TRANSPORT': 'PURPOSE_GOODS',
    'MAIN_USE_SERVICE': 'PURPOSE_SERVICE',
    'NOGA_2008': 'BRANCH',
    'wh_tot_cal': 'STATISTICAL_WEIGHT',
    'Quelle_NPA': 'BASE'})

# Extract stops
origins = shipments_df[['OID', 'quelle_km', 'Quelle_NPA']].copy().rename(
    columns={'quelle_km': 'DIST', 'Quelle_NPA': 'NPA'})
destinations = shipments_df[['OID', 'ziel_km', 'Ziel_NPA']].copy().rename(
    columns={'ziel_km': 'DIST', 'Ziel_NPA': 'NPA'})

stops = pd.concat([origins, destinations]).drop_duplicates(ignore_index=True)
stops = stops.sort_values(['OID', 'DIST'])

# Check whether some stops have the same OID and km but different NPA
stops_check = stops.drop_duplicates(subset=['OID', 'DIST'], ignore_index=True)
if len(stops_check) != len(stops):
    print('Some stops have the same OID and mileage but different NPAs. We sort them randomly.')

# Define tour IDs
stops = stops.merge(oids[['OID', 'BASE']], on='OID')  # adds 'BASE' to the stops dataframe
stops['TOUR_ID'] = np.cumsum(stops['NPA'] == stops['BASE']) - 1  # generate tour IDs

print(len(stops), 'stops')

# Define the set of trip legs (first a too big set)
leg_origins = stops.loc[stops.index[0:-1]].reset_index(drop=True)
leg_destinations = stops.loc[stops.index[1:]].reset_index(drop=True)
legs = leg_origins.join(leg_destinations, lsuffix='_ORIG', rsuffix='_DEST')

# Then drop legs spanning over different surveys
legs = legs[legs['OID_ORIG'] == legs['OID_DEST']]
legs.index = np.arange(len(legs))

# Trip legs:rename some columns
legs = legs.rename(columns={
    'OID_ORIG': 'OID',
    'TOUR_ID_ORIG': 'TOUR_ID',
    'BASE_ORIG': 'BASE',
    'NPA_ORIG': 'ORIG',
    'NPA_DEST': 'DEST'})

# Add trip IDs (i.e. leg rank, starting from 0)
legs['TRIP_ID'] = 0
for i, row in enumerate(legs.to_dict('records')):
    if i > 0:
        if legs.at[i, 'TOUR_ID'] != legs.at[i - 1, 'TOUR_ID']:
            legs.at[i, 'TRIP_ID'] = 0
        else:
            legs.at[i, 'TRIP_ID'] = legs.at[i - 1, 'TRIP_ID'] + 1

legs['DIST_SURVEY'] = legs['DIST_DEST']-legs['DIST_ORIG']
# Keep only column we want to export to CSV
legs = legs[['OID', 'TOUR_ID', 'TRIP_ID', 'ORIG', 'DEST','DIST_SURVEY']]

# Initialize fields for other trip leg attributes
legs[['TTC', 'DIST']] = -1.0
legs[['PURPOSE_GOODS', 'PURPOSE_SERVICE']] = 0
legs['CURB_WEIGHT'] = 0
legs['BRANCH'] = 0
legs['STATISTICAL_WEIGHT'] = 0.0
legs['N_TRIPS'] = 0

# Now fill these fields
for i, row in enumerate(legs.to_dict('records')):

    # Try to find the travel time and distance of the trip in the skim matrix, otherwise keep -1
    i_matrix = zone_mapping.get(row['ORIG'])
    j_matrix = zone_mapping.get(row['DEST'])
    if i_matrix is not None and j_matrix is not None:
        legs.at[i, 'TTC'] = tt_matrix[i_matrix, j_matrix]
        legs.at[i, 'DIST'] = dist_matrix[i_matrix, j_matrix]

    # Purpose, curb weight, branch, statistical weight
    legs.at[i, 'PURPOSE_GOODS'] = oid_purpose_goods[row['OID']] if not pd.isna(oid_purpose_goods[row['OID']]) else 0
    legs.at[i, 'PURPOSE_SERVICE'] = oid_purpose_service[row['OID']] if not pd.isna(oid_purpose_service[row['OID']]) else 0
    legs.at[i, 'CURB_WEIGHT'] = oid_curb_weight[row['OID']]
    legs.at[i, 'BRANCH'] = oid_branch[row['OID']]
    legs.at[i, 'STATISTICAL_WEIGHT'] = oid_statistical_weight[row['OID']]

#%% Data cleaning: remove internal trips where the recorded distance is 4 times larger than in the skim matrix.
legs = legs[~((legs['ORIG'] == legs['DEST']) & (4 * legs['DIST'] <= legs['DIST_SURVEY']))]
legs = legs.reset_index(drop=True)
print(len(legs), 'legs')

#%% Define tours
# Find the rows of the legs in each tour
where_tour_id: Dict[int, List[int]] = {}
for i, row in enumerate(legs.to_dict('records')):
    try:
        where_tour_id[row['TOUR_ID']].append(i)
    except KeyError:
        where_tour_id[row['TOUR_ID']] = [i]

for i, row in enumerate(legs.to_dict('records')):
    # Number of trips in tour
    legs.at[i, 'N_TRIPS'] = len(where_tour_id[row['TOUR_ID']])

# Calculate some extra statistics for every tour
tours: List[List[Any]] = []

for tour_id, leg_indices in where_tour_id.items():
    oid = legs.at[leg_indices[0], 'OID']

    n_legs = len(leg_indices)
    tour_is_internal = int(legs.at[leg_indices[0], 'ORIG'] == legs.at[leg_indices[0], 'DEST'])
    return_trip_made = int(legs.at[leg_indices[0], 'ORIG'] == legs.at[leg_indices[-1], 'DEST']) if not tour_is_internal else -1
    recorded_dist_first_leg = legs.at[leg_indices[0], 'DIST_SURVEY']

    if return_trip_made:
        orig_return = legs.at[leg_indices[-1], 'ORIG']
        dest_return = legs.at[leg_indices[-1], 'DEST']
    else:
        orig_return = legs.at[leg_indices[-1], 'DEST']
        dest_return = legs.at[leg_indices[0], 'ORIG']

    i_matrix = zone_mapping.get(orig_return)
    j_matrix = zone_mapping.get(dest_return)

    if i_matrix is None or j_matrix is None:
        ttc = -1
        dist = -1
    else:
        ttc_return = tt_matrix[i_matrix, j_matrix]
        dist_return = dist_matrix[i_matrix, j_matrix]

    tours.append([
        oid, tour_id, n_legs,
        tour_is_internal, return_trip_made,
        recorded_dist_first_leg,
        ttc_return, dist_return,
        oid_purpose_goods[oid], oid_purpose_service[oid],
        oid_curb_weight[oid], oid_branch[oid],
        oid_statistical_weight[oid]])

tours_df = pd.DataFrame(
    np.array(tours),
    columns=[
        'OID', 'TOUR_ID', 'N_TRIPS',
        'INTERNAL', 'RETURN_TRIP_MADE',
        'RECORDED_DIST_FIRST_LEG',
        'TTC_RETURN', 'DIST_RETURN',
        'PURPOSE_GOODS', 'PURPOSE_SERVICE',
        'CURB_WEIGHT', 'BRANCH', 'STATISTICAL_WEIGHT'])


#%% Data cleaning


#%%

print('Writing shipments, legs and tours to csv...')

shipments_df.to_csv(folder_project / 'data' / 'shipments_lcv_data.csv', sep=',', index=False)
legs.to_csv(folder_project / 'data' / 'trips_lcv_data.csv', sep=',', index=False)
tours_df.to_csv(folder_project / 'data' / 'tours_lcv_data.csv', sep=',', index=False)


#%%

print('Writing legs to shp...')

legs_geometry = [
    (centroids.get(legs.at[i, 'ORIG']), centroids.get(legs.at[i, 'DEST']))
    for i in range(len(legs))]

# Find tours for which we could not get the coordinates of one of the trips
tour_has_missing_geometry = {}
for i, row in enumerate(legs.to_dict('records')):
    tour_id = row['TOUR_ID']
    if tour_has_missing_geometry.get(tour_id) != True:
        tour_has_missing_geometry[tour_id] = (
            legs_geometry[i][0] is None or
            legs_geometry[i][1] is None)

tours_without_missing_geometry = [tour_id for tour_id, value in tour_has_missing_geometry.items() if not value]
legs_without_missing_geometry = [i for i, row in enumerate(legs.to_dict('records')) if not tour_has_missing_geometry[row['TOUR_ID']]]

legs_shp = legs.iloc[legs_without_missing_geometry, :].copy()

legs_shp['X_ORIG'] = [centroids[x][0] for x in legs_shp['ORIG'].values]
legs_shp['Y_ORIG'] = [centroids[x][1] for x in legs_shp['ORIG'].values]
legs_shp['X_DEST'] = [centroids[x][0] for x in legs_shp['DEST'].values]
legs_shp['Y_DEST'] = [centroids[x][1] for x in legs_shp['DEST'].values]

write_shape_line(
    str(folder_project / 'data' / 'trips_lcv_data.shp'),
    legs_shp,
    ['X_ORIG', 'Y_ORIG', 'X_DEST', 'Y_DEST'])

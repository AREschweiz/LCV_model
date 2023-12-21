import pandas as pd
from pathlib import Path

folder_project = Path.cwd().parent

path_outputs = folder_project / 'outputs' / 'parcel_delivery'

path_params_parcel_couriers = folder_project / 'parameters' / 'ParcelCouriers.csv'
path_params_parcel_depots = folder_project / 'parameters' / 'ParcelDepots.csv'
path_zone_stats = folder_project / 'data' / 'zone_stats_2013_NPVM.csv'

max_parcels_per_van = 160
parcels_per_person = 0.0752 * 1.23
parcels_per_job = 0.0471 * 1.23

print('Reading files...')

courier_shares = dict(
    (row['Courier'], row['Share'])
    for row in pd.read_csv(path_params_parcel_couriers, sep=';').to_dict('records'))

parcel_depots = pd.read_csv(path_params_parcel_depots, sep=';')

zone_stats = pd.read_csv(path_zone_stats, sep=';')

jobs = (zone_stats['Industrial'] + zone_stats['Services'] + zone_stats['Trade'] + zone_stats['Transport']).values
population = zone_stats['Pop'].values

zone_mapping = dict((value, i) for i, value in enumerate(zone_stats['Zone NPVM'].values))

parcel_demand = pd.read_csv(path_outputs / 'parcel_demand.csv', sep=';')
parcel_schedules = pd.read_csv(path_outputs / 'parcel_schedules.csv', sep=';')

print('Calculating statistics...')

parcels_per_courier = {
    'demand': pd.pivot_table(parcel_demand, index='courier', values='n_parcels', aggfunc=sum).to_dict()['n_parcels'],
    'schedules': pd.pivot_table(parcel_schedules, index='courier', values='n_parcels', aggfunc=sum).to_dict()['n_parcels']}

parcels_per_depot = {
    'demand': pd.pivot_table(parcel_demand, index='depot_id', values='n_parcels', aggfunc=sum).to_dict()['n_parcels'],
    'schedules': pd.pivot_table(parcel_schedules, index='depot_id', values='n_parcels', aggfunc=sum).to_dict()['n_parcels']}

parcels_per_dest_zone = {
    'demand': pd.pivot_table(parcel_demand, index='dest_zone', values='n_parcels', aggfunc=sum).to_dict()['n_parcels'],
    'schedules': pd.pivot_table(parcel_schedules, index='dest_zone', values='n_parcels', aggfunc=sum).to_dict()['n_parcels']}

parcels_per_tour = pd.pivot_table(parcel_schedules, index='tour_id', values='n_parcels', aggfunc=sum)

print('Performing checks...')

# Compare realized market shares of couriers with input
for file_type in ['demand', 'schedules']:

    for courier in courier_shares.keys():

        if parcels_per_courier[file_type].get(courier) is None:
            print(f"Courier '{courier}' not found in 'parcel_{file_type}'.")
        else:
            share_realization = round(parcels_per_courier[file_type][courier] / sum(parcels_per_courier[file_type].values()), 4)
            share_expected = round(courier_shares[courier], 4)
            if share_realization != share_expected:
                print(
                    f"The realized market share of courier '{courier}' ({share_realization}) in 'parcel_{file_type}' " +
                    f"differs from the market share in 'params_parcel_couriers' ({share_expected}).")

# Compare number of parcels per depot between 'demand' and 'schedules'
for depot_id in parcel_depots['DepotID'].values:

    n_parcels_depot_demand = parcels_per_depot['demand'].get(depot_id, 0)
    n_parcels_depot_schedules = parcels_per_depot['schedules'].get(depot_id, 0)

    if n_parcels_depot_demand != n_parcels_depot_schedules:
        print(
            f"The number of parcels for depot_id {depot_id} differs between " +
            f"the 'parcel_demand' ({n_parcels_depot_demand}) and " +
            f"the 'parcel_schedules' ({n_parcels_depot_schedules}).")

# Compare number of parcels per dest_zone between 'demand' and 'schedules'
for dest_zone in zone_mapping.keys():

    n_parcels_dest_zone_demand = parcels_per_dest_zone['demand'].get(dest_zone, 0)
    n_parcels_dest_zone_schedules = parcels_per_dest_zone['schedules'].get(dest_zone, 0)

    if n_parcels_dest_zone_demand != n_parcels_dest_zone_schedules:
        print(
            f"The number of parcels for dest_zone {dest_zone} differs between " +
            f"the 'parcel_demand' ({n_parcels_dest_zone_demand}) and " +
            f"the 'parcel_schedules' ({n_parcels_dest_zone_schedules}).")

# Check on maximum number of parcels per tour
n_tours_with_too_many_parcels = (parcels_per_tour > max_parcels_per_van).sum().sum()
if n_tours_with_too_many_parcels > 0:
    print(f"There are {n_tours_with_too_many_parcels} parcel tours with more than {max_parcels_per_van} parcels.")

# Compare number of parcels per dest_zone with expected number
tolerance = 2.0

for dest_zone in zone_mapping.keys():

    n_parcels_realized = parcels_per_dest_zone['demand'].get(dest_zone, 0)
    n_parcels_expected = round(
        parcels_per_person * population[zone_mapping[dest_zone]] +
        parcels_per_job * jobs[zone_mapping[dest_zone]], 1)

    if abs(n_parcels_realized - n_parcels_expected) > tolerance:
        print(
            f"The calculated number of parcels in dest_zone {dest_zone} ({n_parcels_realized}) " +
            f"differs more than {tolerance} parcels from the expected number ({n_parcels_expected}).")

print('Script finished.')

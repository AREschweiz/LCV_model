import logging
import numpy as np
import pandas as pd

from typing import Any, Dict

from src.support import (
    write_shape_line,
    get_setting, get_input_path, get_output_path,
    get_omx_matrix)

logger = logging.getLogger('logger')


def main(config: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Main function of the Parcel Demand Synthesis calculation module.
    """
    n_external_zones = get_setting(config, 'n_external_zones', int)

    sep = get_setting(config, 'sep', str)
    write_csv = get_setting(config, 'write_csv', bool)
    write_shp = get_setting(config, 'write_shp', bool)

    path_centroids = get_input_path(config, 'centroids')
    path_dist_matrix = get_input_path(config, 'dist_matrix')
    path_zone_neighbors = get_input_path(config, 'zone_neighbors')
    path_zone_stats = get_input_path(config, 'zone_stats')

    path_params_parcel_couriers = get_input_path(config, 'params_parcel_couriers')
    path_params_parcel_demand = get_input_path(config, 'params_parcel_demand')
    path_params_parcel_depots = get_input_path(config, 'params_parcel_depots')

    path_out_parcel_demand_csv = get_output_path(config, 'parcel_demand_csv')
    path_out_parcel_demand_shape = get_output_path(config, 'parcel_demand_shape')

    logger.info("\tImporting data...")

    zone_stats = pd.read_csv(path_zone_stats, sep=sep)
    n_zones = zone_stats.shape[0]
    jobs = dict(
        (row['ID'], row['Jobs'])
        for row in zone_stats.to_dict('records'))
    population = dict(
        (row['ID'], row['Pop'])
        for row in zone_stats.to_dict('records'))

    centroids = pd.read_csv(path_centroids, sep=sep)

    dist_matrix, zone_mapping, zone_ids = get_omx_matrix(path_dist_matrix, n_zones, n_external_zones)

    zone_neighbors = [[] for i in range(n_zones)]
    for row in pd.read_csv(path_zone_neighbors, sep=sep).to_dict('records'):
        zone_neighbors[zone_mapping[row['ZONE_1']]].append(zone_mapping[row['ZONE_2']])

    demand_parameters = get_demand_parameters(path_params_parcel_demand, sep)
    courier_shares = get_courier_shares(path_params_parcel_couriers, sep)
    depots = get_depots(path_params_parcel_depots, courier_shares, sep)

    logger.info("\tCalculating parcel demand per zone and courier...")

    parcel_demand = calc_parcel_demand(
        demand_parameters, courier_shares,
        jobs, population, zone_mapping)

    logger.info("\tAssigning parcels to depots...")

    parcel_demand = assign_parcels_to_depots(
        parcel_demand, depots, dist_matrix, zone_mapping, centroids, zone_neighbors)

    columns_coords = ['orig_x_coord', 'orig_y_coord', 'dest_x_coord', 'dest_y_coord']

    if write_shp:
        logger.info('\tWriting shapefile of parcel demand...')
        write_shape_line(path_out_parcel_demand_shape, parcel_demand, columns_coords)

    if write_csv:
        logger.info('\tWriting parcel demand to .csv...')
        parcel_demand.to_csv(path_out_parcel_demand_csv, sep=sep, index=False)

    return parcel_demand


def get_demand_parameters(
    path_params_parcel_demand: str,
    sep: str
) -> Dict[str, float]:
    """
    Reads the parcel demand parameters as a dictionary.
    """
    demand_parameters = {
        'ParcelsPerPerson': None,
        'ParcelsPerJob': None,
        'AverageNumAttemptsB2C': None,
        'AverageNumAttemptsB2B': None}

    for row in pd.read_csv(path_params_parcel_demand, sep=sep).to_dict('records'):
        demand_parameters[row['Parameter']] = row['Value']

    for key, value in demand_parameters.items():
        if value is None:
            raise Exception(f"Could not obtain a value for parameter '{key}' from 'params_parcel_demand'.")
        logger.debug(f"\t\t{key}: {value}")

    return demand_parameters


def get_courier_shares(
    path_params_parcel_couriers: str,
    sep: str
) -> Dict[str, float]:
    """
    Gets the market share of each parcel courier as a dictionary.
    """
    courier_shares = {}

    for row in pd.read_csv(path_params_parcel_couriers, sep=sep).to_dict('records'):
        courier_name = row['Courier']
        market_share = row['Share']
        courier_shares[courier_name] = market_share

        if type(row['Share']) is not float:
            raise Exception(
                f"The 'Share' {market_share} for 'Courier' {courier_name} is not a float value.")

    sum_market_share = round(sum(courier_shares.values()), 3)

    if sum_market_share != 1.0:
        raise Exception(
            f"The sum of 'Share' ({sum_market_share}) in 'params_parcel_couriers' should equal 1.0.")

    return courier_shares


def get_depots(
    path_params_parcel_depots: str,
    courier_shares: Dict[str, float],
    sep: str
) -> pd.DataFrame:
    """
    Gets the parcel depots and checks whether its listed 'Courier' values are found in
    table 'params_parcel_couriers' too and vice versa.
    """
    depots = pd.read_csv(path_params_parcel_depots, sep=sep)[['DepotID', 'Courier', 'ZoneID']]
    depots_couriers_unique = np.unique(depots['Courier'].values)

    for courier_name, share in courier_shares.items():
        if share > 0.0:
            if courier_name not in depots_couriers_unique:
                raise Exception(
                    f"Parcel courier '{courier_name}' is found in 'params_parcel_couriers'" +
                    " but not in 'params_parcel_depots'.")

    for courier_name in depots_couriers_unique:
        if courier_name not in courier_shares.keys():
            raise Exception(
                f"Parcel courier '{courier_name}' is found in 'params_parcel_depots'" +
                " but not in 'params_parcel_couriers'.")

    return depots


def calc_parcel_demand(
    demand_parameters: Dict[str, float],
    courier_shares: Dict[str, float],
    jobs: Dict[int, float],
    population: Dict[int, float],
    zone_mapping: Dict[int, int]
) -> pd.DataFrame:
    """
    Calculates the number of parcels to be delivered in each zone by each courier.
    """
    parcel_demand = []

    for zone_id in zone_mapping.keys():
        n_parcels_zone = (
            demand_parameters['AverageNumAttemptsB2B'] * demand_parameters['ParcelsPerJob'] * jobs.get(zone_id, 0.0) +
            demand_parameters['AverageNumAttemptsB2C'] * demand_parameters['ParcelsPerPerson'] * population.get(zone_id, 0.0))

        for courier_name, share in courier_shares.items():
            parcel_demand.append([zone_id, courier_name, n_parcels_zone * share])

    parcel_demand = pd.DataFrame(parcel_demand, columns=['dest_zone', 'courier', 'n_parcels'])

    parcel_demand['n_parcels'] = parcel_demand['n_parcels'].round().astype(int)

    parcel_demand = parcel_demand[parcel_demand['n_parcels'] > 0.0]

    return parcel_demand


def assign_parcels_to_depots(
    parcel_demand: pd.DataFrame,
    depots: pd.DataFrame,
    dist_matrix: np.ndarray,
    zone_mapping: Dict[int, int],
    centroids: pd.DataFrame,
    zone_neighbors: np.ndarray
) -> pd.DataFrame:
    """
    Determine for each zone and courier which depot will be used for the delivery.
    Also add coordinates of origin and destination during the process.
    """
    zone_mapping_inv = dict((v, k) for k, v in zone_mapping.items())
    n_zones = len(zone_mapping)

    parcel_demand_courier = parcel_demand['courier'].values
    parcel_demand_dest_zone = np.array(
        [zone_mapping[zone_id] for zone_id in parcel_demand['dest_zone'].values], dtype=int)
    depots_zone = np.array(
        [zone_mapping[zone_id] for zone_id in depots['ZoneID'].values], dtype=int)

    # Choose the nearest depot of the courier as the initial depot for every parcel flow
    parcel_demand_depot = np.zeros(parcel_demand.shape[0], dtype=int)

    for i in range(parcel_demand.shape[0]):
        where_courier_depot = np.where(depots['Courier'] == parcel_demand_courier[i])[0]
        parcel_demand_depot[i] = where_courier_depot[
            np.argmin(dist_matrix[depots_zone[where_courier_depot], parcel_demand_dest_zone[i]])]

    # Improve the depot allocation for "isolated" zones, i.e. flows that cannot form a path back to the depot
    # through the other zones allocated to this depot
    # (Do this in three iterations)
    for i in range(3):
        parcel_demand_depot = improve_depot_assignment(
            depots, depots_zone, parcel_demand_courier, parcel_demand_depot, parcel_demand_dest_zone,
            zone_neighbors, dist_matrix, n_zones)

    parcel_demand['depot_id'] = parcel_demand_depot + 1

    parcel_demand_orig_zone = depots_zone[parcel_demand_depot]
    parcel_demand['orig_zone'] = [zone_mapping_inv[zone] for zone in parcel_demand_orig_zone]

    parcel_demand['orig_x_coord'] = [centroids.at[zone, 'x_coord'] for zone in parcel_demand_orig_zone]
    parcel_demand['orig_y_coord'] = [centroids.at[zone, 'y_coord'] for zone in parcel_demand_orig_zone]
    parcel_demand['dest_x_coord'] = [centroids.at[zone, 'x_coord'] for zone in parcel_demand_dest_zone]
    parcel_demand['dest_y_coord'] = [centroids.at[zone, 'y_coord'] for zone in parcel_demand_dest_zone]

    return parcel_demand


def improve_depot_assignment(
    depots: pd.DataFrame,
    depots_zone: np.ndarray,
    parcel_demand_courier: np.ndarray,
    parcel_demand_depot: np.ndarray,
    parcel_demand_dest_zone: np.ndarray,
    zone_neighbors: np.ndarray,
    dist_matrix: np.ndarray,
    n_zones: int
):
    """
    Improve the depot assignment for cases of "isolated zones", i.e., no path back to the depot can be formed
    through the other zones assigned to this depot.
    """
    for depot_id, row in enumerate(depots.to_dict('records')):

        where_courier_depot = np.where(depots['Courier'] == row['Courier'])[0]

        # If the courier only has one depot, we cannot make any switches in terms of depot allocation
        if len(where_courier_depot) == 1:
            continue

        # To which depot is each of the zones for the courier of the current depot assigned
        zone_assignment = - np.ones(n_zones, dtype=int)
        for i in np.where(parcel_demand_courier == row['Courier'])[0]:
            zone_assignment[parcel_demand_dest_zone[i]] = parcel_demand_depot[i]

        # Check which zones can form a path back to the depot through the other zones allocated to this depot
        zone_connects_to_depot = np.zeros(n_zones, dtype=int)

        def find_connected_zones(current_zone: int):
            for neighbor_zone in zone_neighbors[current_zone]:
                if zone_connects_to_depot[neighbor_zone] != 1:
                    if zone_assignment[neighbor_zone] == zone_assignment[current_zone]:
                        zone_connects_to_depot[neighbor_zone] = 1
                        find_connected_zones(neighbor_zone)

        find_connected_zones(depots_zone[depot_id])

        isolated_zones = [
            i for i in range(n_zones)
            if zone_connects_to_depot[i] == 0 and zone_assignment[i] == depot_id]

        # Check if we can improve this by choosing another depot for these "isolated" zones
        for isolated_zone in isolated_zones:
            where_parcels_to_zone = np.where(
                (parcel_demand_depot == depot_id) &
                (parcel_demand_dest_zone == isolated_zone))[0]

            distances_depots = dist_matrix[depots_zone[where_courier_depot], isolated_zone]
            argsort_depots = np.argsort(distances_depots)

            optional_depot_1 = argsort_depots[1]

            optional_depot_1_connects = np.any(
                zone_assignment[zone_neighbors[isolated_zone]] ==
                where_courier_depot[optional_depot_1])

            # Assign flow to the second nearest depot if it resolves the isolation
            if optional_depot_1_connects:
                parcel_demand_depot[where_parcels_to_zone] = where_courier_depot[optional_depot_1]
                zone_assignment[isolated_zone] = where_courier_depot[optional_depot_1]

            # Otherwise, assign flow to the third nearest depot if it resolves the isolation
            elif len(where_courier_depot) >= 3:
                optional_depot_2 = argsort_depots[2]

                optional_depot_2_connects = np.any(
                    zone_assignment[zone_neighbors[isolated_zone]] ==
                    where_courier_depot[optional_depot_2])

                if optional_depot_2_connects:
                    parcel_demand_depot[where_parcels_to_zone] = where_courier_depot[optional_depot_2]
                    zone_assignment[isolated_zone] = where_courier_depot[optional_depot_2]

    return parcel_demand_depot

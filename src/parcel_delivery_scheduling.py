import logging
import numpy as np
import pandas as pd

from typing import Any, Dict, List, Tuple

from src.support import (
    write_shape_line,
    get_setting, get_input_path, get_output_path,
    get_omx_matrix,
    improve_tour_sequence)

logger = logging.getLogger('logger')


def main(
    config: Dict[str, Dict[str, Any]],
    parcel_demand: pd.DataFrame
) -> pd.DataFrame:
    """
    Main function of the Parcel Delivery Scheduling calculation module.
    """
    n_external_zones = get_setting(config, 'n_external_zones', int)

    sep = get_setting(config, 'sep', str)
    write_csv = get_setting(config, 'write_csv', bool)
    write_shp = get_setting(config, 'write_shp', bool)

    path_centroids = get_input_path(config, 'centroids')
    path_dist_matrix = get_input_path(config, 'dist_matrix')
    path_tt_matrix = get_input_path(config, 'tt_matrix')

    path_params_parcel_depots = get_input_path(config, 'params_parcel_depots')
    path_params_parcel_scheduling = get_input_path(config, 'params_parcel_scheduling')

    path_out_parcel_schedules_csv = get_output_path(config, 'parcel_schedules_csv')
    path_out_parcel_schedules_shape = get_output_path(config, 'parcel_schedules_shape')

    logger.info("\tImporting data...")

    centroids = pd.read_csv(path_centroids, sep=sep)
    n_zones = centroids.shape[0] - n_external_zones

    tt_matrix, zone_mapping, zone_ids = get_skim_matrix(path_tt_matrix, n_zones, n_external_zones)
    dist_matrix, zone_mapping, zone_ids = get_skim_matrix(path_dist_matrix, n_zones, n_external_zones)

    depots = pd.read_csv(path_params_parcel_depots, sep=sep)[['DepotID', 'Courier', 'ZoneID']]

    parcels_per_van, max_parcel_tour_duration = get_params_parcel_scheduling(path_params_parcel_scheduling, sep)

    logger.info("\tForming delivery clusters...")

    rearranged_parcel_demand = rearrange_parcel_demand(parcel_demand)

    clustered_parcel_demand = cluster_parcels(
        rearranged_parcel_demand, parcels_per_van, dist_matrix, tt_matrix, zone_mapping,
        max_parcel_tour_duration * 60)

    logger.info("\tConstructing tour schedules...")

    parcel_schedules = construct_parcel_schedules(
        clustered_parcel_demand, dist_matrix, zone_mapping,
        centroids, depots)

    columns_coords = ['orig_x_coord', 'orig_y_coord', 'dest_x_coord', 'dest_y_coord']

    if write_shp:
        logger.info('\tWriting shapefile of parcel schedules...')
        write_shape_line(path_out_parcel_schedules_shape, parcel_schedules, columns_coords)

    if write_csv:
        logger.info('\tWriting parcel schedules to .csv...')
        parcel_schedules.to_csv(path_out_parcel_schedules_csv, sep=sep, index=False)

    return parcel_schedules


def get_params_parcel_scheduling(
    path_params_percel_scheduling: str,
    sep: str
) -> Tuple[int, int]:
    """
    Get the vehicle capacity as number of parcels and maximum parcel tour duration in hours.
    """
    parcels_per_van: int = None
    max_parcel_tour_duration: int = None

    for row in pd.read_csv(path_params_percel_scheduling, sep=sep).to_dict('records'):

        if row['Parameter'] == 'ParcelsPerVan':
            try:
                parcels_per_van = int(row['Value'])
            except ValueError:
                raise Exception(f"Could not convert '{row['Value']}' in 'params_parcel_scheduling' to an integer.")

        if row['Parameter'] == 'MaxParcelTourDuration':
            try:
                max_parcel_tour_duration = int(row['Value'])
            except ValueError:
                raise Exception(f"Could not convert '{row['Value']}' in 'params_parcel_scheduling' to an integer.")

    if parcels_per_van is None:
        raise Exception("Could not find 'ParcelsPerVan' in 'params_parcel_scheduling'.")

    if max_parcel_tour_duration is None:
        raise Exception("Could not find 'MaxParcelTourDuration' in 'params_parcel_scheduling'.")

    logger.debug(f'\t\tparcels_per_van: {parcels_per_van}')
    logger.debug(f'\t\tmax_parcel_tour_duration: {max_parcel_tour_duration}')

    return (parcels_per_van, max_parcel_tour_duration)


def rearrange_parcel_demand(
    parcel_demand: pd.DataFrame
) -> pd.DataFrame:
    """
    Rearrange the parcel demand such that each row is one parcel rather than a group of parcels.
    """
    parcel_demand_rearranged = []

    for row in parcel_demand.to_dict('records'):
        row_to_repeat = [row['depot_id'], row['orig_zone'], row['dest_zone']]
        for i in range(row['n_parcels']):
            parcel_demand_rearranged.append(row_to_repeat.copy())

    parcel_demand_rearranged = pd.DataFrame(
        np.array(parcel_demand_rearranged),
        columns=['depot_id', 'orig_zone', 'dest_zone'])

    return parcel_demand_rearranged


def cluster_parcels(
    rearranged_parcel_demand: pd.DataFrame,
    parcels_per_van: int,
    dist_matrix: np.ndarray,
    tt_matrix: np.ndarray,
    zone_mapping: Dict[int, int],
    max_tour_duration_minutes: float
) -> pd.DataFrame:
    """
    Assign parcels to clusters based on spatial proximity with cluster size constraints.
    The cluster variable is added as extra column to the DataFrame.
    """
    depot_ids: List[int] = np.unique(rearranged_parcel_demand['depot_id'])

    rearranged_parcel_demand.index = np.arange(rearranged_parcel_demand.shape[0])

    parcels_cluster: np.ndarray = - np.ones(rearranged_parcel_demand.shape[0])

    # First check for depot/destination combination with more than 'parcels_per_van' parcels.
    # For these we don't need to use the clustering algorithm.
    counts = pd.pivot_table(
        rearranged_parcel_demand, index=['depot_id', 'dest_zone'], aggfunc=len)
    where_large_cluster = list(counts.index[np.where(counts >= parcels_per_van)[0]])

    parcels_depot_id: np.ndarray = np.array(rearranged_parcel_demand['depot_id'], dtype=int)
    parcels_dest_zone: np.ndarray = np.array(rearranged_parcel_demand['dest_zone'], dtype=int)

    where_depot_dest: Dict[Tuple[int, int], List[int]] = {}
    for i in range(rearranged_parcel_demand.shape[0]):
        try:
            where_depot_dest[(parcels_depot_id[i], parcels_dest_zone[i])].append(i)
        except KeyError:
            where_depot_dest[(parcels_depot_id[i], parcels_dest_zone[i])] = [i]

    cluster_id: int = 0

    for depot_id, dest_zone in where_large_cluster:

        indices = where_depot_dest[(depot_id, dest_zone)]

        for i in range(int(np.floor(len(indices) / parcels_per_van))):
            parcels_cluster[indices[:parcels_per_van]] = cluster_id
            indices = indices[parcels_per_van:]

            cluster_id += 1

    clustered_parcel_demand: pd.DataFrame = rearranged_parcel_demand.copy()
    clustered_parcel_demand['cluster'] = parcels_cluster

    # For each depot, cluster remaining parcels into batches of 'parcels_per_van' parcels
    for depot_id in depot_ids:

        # Select parcels of the depot that are not assigned a cluster yet
        parcels_to_fit: pd.DataFrame = clustered_parcel_demand[
            (clustered_parcel_demand['depot_id'] == depot_id) &
            (clustered_parcel_demand['cluster'] == -1)].copy()

        parcels_to_fit['orig_zone'] = [zone_mapping[zone] for zone in parcels_to_fit['orig_zone'].values]
        parcels_to_fit['dest_zone'] = [zone_mapping[zone] for zone in parcels_to_fit['dest_zone'].values]

        # Sort parcels descending based on distance to depot so that at the end of the loop the remaining parcels
        # are all nearby the depot and form a somewhat reasonable parcels cluster
        parcels_to_fit['distance'] = dist_matrix[parcels_to_fit['orig_zone'], parcels_to_fit['dest_zone']]
        parcels_to_fit = parcels_to_fit.sort_values('distance', ascending=False)
        parcels_to_fit_index = list(parcels_to_fit.index)
        parcels_to_fit.index = np.arange(len(parcels_to_fit))
        dests = np.array(parcels_to_fit['dest_zone'])

        # How many tours are needed to deliver these parcels
        n_tours_needed = int(np.ceil(len(parcels_to_fit) / parcels_per_van))

        # In the case of 1 tour it's simple, all parcels belong to the same cluster
        if n_tours_needed == 1:
            clustered_parcel_demand.loc[parcels_to_fit_index, 'cluster'] = cluster_id
            cluster_id += 1
            continue

        # When there are multiple tours needed, the heuristic is a little bit more complex
        clusters = np.ones(len(parcels_to_fit), dtype=int) * -1

        tour_orig_zone: int = parcels_to_fit['orig_zone'].values[0]

        # Keep assigning parcels to clusters until they are all assigned
        while True:

            # Select the first parcel for the new cluster that is now initialized
            yet_assigned = (clusters != -1)
            not_yet_assigned = np.where(~yet_assigned)[0]

            if len(not_yet_assigned) == 0:
                break

            first_parcel_index = not_yet_assigned[0]

            clusters[first_parcel_index] = cluster_id
            tour_dest_zones: List[int] = [dests[first_parcel_index]]

            # Find the nearest {max_vehicle_load - 1} parcels to this first parcel that are not in a cluster yet
            distances = dist_matrix[dests[first_parcel_index], dests]
            distances[not_yet_assigned[0]] = 99999
            distances[yet_assigned] = 99999
            where_assign_cluster_id = np.argsort(distances)[:(parcels_per_van - 1)]

            clusters[where_assign_cluster_id[0]] = cluster_id
            tour_dest_zones.append(dests[where_assign_cluster_id[0]])

            for i in range(1, parcels_per_van - 1):

                if dests[where_assign_cluster_id[i]] != dests[where_assign_cluster_id[i - 1]]:
                    tour_dest_zones.append(dests[where_assign_cluster_id[i]])
                    tour_sequence = construct_tour_sequence(tour_orig_zone, tour_dest_zones, dist_matrix)
                    tour_duration: float = np.sum(tt_matrix[tour_sequence[:-1], tour_sequence[1:]])

                    if tour_duration > max_tour_duration_minutes:
                        break

                clusters[where_assign_cluster_id[i]] = cluster_id

            cluster_id += 1

        clustered_parcel_demand.loc[parcels_to_fit_index, 'cluster'] = clusters

    clustered_parcel_demand['cluster'] = clustered_parcel_demand['cluster'].astype(int)

    # Check if all parcels are indeed assigned to a cluster
    if np.any(clustered_parcel_demand['cluster'] == -1):
        n_unassigned_parcels: int = np.sum(clustered_parcel_demand['cluster'] == -1)
        logger.warning(f"There are {n_unassigned_parcels} parcels not assigned to a cluster.")

    return clustered_parcel_demand


def construct_tour_sequence(
    orig_zone: int,
    dest_zones: List[int],
    dist_matrix: np.ndarray
) -> List[int]:
    """
    Constructs a sequence of visiting zones, given an origin zone and destination zones to be visited.
    """
    n_stops = 2 + len(dest_zones)

    # Apply nearest neighbor search
    current_zone = orig_zone
    unvisited_dest_zones = dest_zones.copy()

    sequence_init = [orig_zone]
    while len(sequence_init) < (n_stops - 1):
        argmin_ind = np.argmin(dist_matrix[current_zone, unvisited_dest_zones])
        current_zone = unvisited_dest_zones[argmin_ind]
        unvisited_dest_zones.pop(argmin_ind)
        sequence_init.append(current_zone)
    sequence_init.append(orig_zone)

    if n_stops <= 4:
        return sequence_init

    # Apply swapping algorithm for improvement
    sequence_post = improve_tour_sequence(np.array(sequence_init, dtype=int), dist_matrix)

    return sequence_post


def construct_parcel_schedules(
    clustered_parcel_demand: pd.DataFrame,
    dist_matrix: np.ndarray,
    zone_mapping: Dict[int, int],
    centroids: Dict[int, Tuple[float, float]],
    depots: pd.DataFrame
) -> pd.DataFrame:
    """
    Construct the schedules for all clusters.
    """
    zone_mapping_inv = dict((v, k) for k, v in zone_mapping.items())

    where_cluster = {}
    for i, row in enumerate(clustered_parcel_demand.to_dict('records')):
        cluster = row['cluster']
        try:
            where_cluster[cluster].append(i)
        except KeyError:
            where_cluster[cluster] = [i]

    centroids_dict = dict(
        (centroids.at[i, 'zone_id'], tuple(centroids.loc[i, ['x_coord', 'y_coord']]))
        for i in centroids.index)

    depot_to_courier = dict(
        (depots.at[i, 'DepotID'], depots.at[i, 'Courier'])
        for i in depots.index)

    tours = []

    tour_id = 0

    for cluster_indices in where_cluster.values():
        current_parcels = clustered_parcel_demand.loc[cluster_indices, :].to_dict('records')

        depot_id = current_parcels[0]['depot_id']
        orig_zone = zone_mapping[current_parcels[0]['orig_zone']]
        dest_zones = [
            zone_mapping[dest_zone]
            for dest_zone in np.unique([row['dest_zone'] for row in current_parcels])]

        n_parcels_per_dest = {}
        for row in current_parcels:
            dest_zone = zone_mapping[row['dest_zone']]
            try:
                n_parcels_per_dest[dest_zone] += 1
            except KeyError:
                n_parcels_per_dest[dest_zone] = 1

        tour_sequence = construct_tour_sequence(orig_zone, dest_zones, dist_matrix)

        for trip_id in range(len(tour_sequence) - 1):
            tmp_orig_zone = zone_mapping_inv[tour_sequence[trip_id]]
            tmp_dest_zone = zone_mapping_inv[tour_sequence[trip_id + 1]]

            tmp_coords = [*centroids_dict[tmp_orig_zone], *centroids_dict[tmp_dest_zone]]

            tmp_n_parcels = (
                0 if trip_id == len(tour_sequence) - 2
                else n_parcels_per_dest.get(tour_sequence[trip_id + 1], 0))

            tours.append([
                depot_id, tour_id, trip_id,
                tmp_orig_zone, tmp_dest_zone,
                tmp_n_parcels,
                *tmp_coords])

        tour_id += 1

    tours = pd.DataFrame(
        np.array(tours),
        columns=[
            'depot_id', 'tour_id', 'trip_id', 'orig_zone', 'dest_zone', 'n_parcels',
            'orig_x_coord', 'orig_y_coord', 'dest_x_coord', 'dest_y_coord'])

    tours['courier'] = [depot_to_courier[depot] for depot in tours['depot_id'].values]

    return tours

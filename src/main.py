# Main script for the simulation of LCV-model (agent-based version)
# Authors:
# - Raphael Ancel, Swiss Federal Office for Spatial Development
# - Sebastiaan Thoen, Significance BV

import functools
import logging
import multiprocessing as mp
import numpy as np
import openmatrix as omx
import pandas as pd
import time

from numba import njit
from typing import Any, Dict, List, Tuple

from src.support import (
    read_yaml, get_logger, log_and_check_config,
    write_shape_line,
    get_dimension, get_setting, get_input_path, get_output_path, get_omx_matrix, get_n_cpu,
    improve_tour_sequence, get_zone_stats)

from parcel_delivery_model import (parcel_demand_synthesis, parcel_delivery_scheduling)


def main(config: Dict[str, Dict[str, Any]], logger: logging.Logger):
    max_tour_duration_minutes = get_setting(config, 'max_tour_duration', float) * 60
    min_n_stops_for_2_opt = get_setting(config, 'min_n_stops_for_2_opt', int)
    n_external_zones = get_setting(config, 'n_external_zones', int)

    # The granularity defines how "big" the modelled agents are.
    # A granularity of 0.1 means that if in reality 100 tours are generated from a zone on a representative day,
    # then 100/0.1= 1000 tours are generated in simulations, each of them counting for 0.1 vehicle equivalent.
    # The smaller the granularity, the longer the computational time and the smaller the variability between simulations.
    granularity = get_setting(config, 'granularity', float)

    sep = get_setting(config, 'sep', str)
    write_omx = get_setting(config, 'write_omx', bool)
    write_csv = get_setting(config, 'write_csv', bool)
    write_shp = get_setting(config, 'write_shp', bool)
    run_parcel_module = get_setting(config, 'run_parcel_module', bool)
    weekday = get_setting(config, 'weekday', bool)

    path_centroids = get_input_path(config, 'centroids')
    path_dist_matrix = get_input_path(config, 'dist_matrix')
    path_tt_matrix = get_input_path(config, 'tt_matrix')
    path_zone_stats = get_input_path(config, 'zone_stats')
    path_mapping_PLZ = get_input_path(config, 'mapping_PLZ')

    path_params_end_tour = get_input_path(config, 'params_end_tour')
    path_params_next_stop = get_input_path(config, 'params_next_stop')
    path_params_n_tours = get_input_path(config, 'params_n_tours')
    path_params_share_active = get_input_path(config, 'params_share_active')
    path_params_vehicle_generation = get_input_path(config, 'params_vehicle_generation')
    path_daily_dist_per_branch_empirical_weekday = get_input_path(config, 'daily_dist_per_branch_empirical_weekday')
    path_daily_dist_per_branch_empirical_full_week = get_input_path(config, 'daily_dist_per_branch_empirical_full_week')

    path_out_trips_csv = get_output_path(config, 'trips_csv')
    path_out_trips_shape = get_output_path(config, 'trips_shape')
    path_out_trip_matrix_csv = get_output_path(config, 'trip_matrix_csv')
    path_out_trip_matrix_omx = get_output_path(config, 'trip_matrix_omx')
    path_out_daily_dist_per_branch = get_output_path(config, 'daily_dist_per_branch_csv')

    dim_branch = get_dimension(config, 'branch')
    dim_segment = get_dimension(config, 'segment')

    segment_ids = [d['id'] for d in dim_segment]
    segment_inds = dict(zip(segment_ids, range(0, len(segment_ids))))

    n_segment = len(dim_segment)
    n_branches = len(dim_branch)

    if run_parcel_module:
        logger.info('Parcel Demand Synthesis module...')

        parcel_demand = parcel_demand_synthesis.main(config)

        logger.info('Parcel Delivery Scheduling module...')

        parcel_schedules = parcel_delivery_scheduling.main(config, parcel_demand)

    logger.info('Generic LCV module...')

    logger.info('\tReading input data...')

    zone_stats, population, jobs, land_use = get_zone_stats(path_zone_stats, sep)
    n_zones = len(zone_stats.index)

    tt_matrix, zone_mapping, zone_ids = get_omx_matrix(path_tt_matrix, n_zones, n_external_zones)
    dist_matrix, zone_mapping, zone_ids = get_omx_matrix(path_dist_matrix, n_zones, n_external_zones)

    inv_zone_mapping = dict((value, key) for key, value in zone_mapping.items())
    mapping_PLZ = pd.read_csv(path_mapping_PLZ, index_col=0, sep=sep)
    PLZ_list = mapping_PLZ['PLZ'].unique()

    # construct a dict with all zones having each PLZ
    PLZ_to_NPVM = {PLZ: [] for PLZ in PLZ_list}
    for zone_NPVM, row in mapping_PLZ.iterrows():
        PLZ_to_NPVM[row['PLZ']].append(zone_NPVM)

    # Fill in same_PLZ_matrix
    same_PLZ_matrix = np.zeros(shape=[n_zones + n_external_zones, n_zones + n_external_zones], dtype=bool)
    for PLZ in PLZ_list:
        for zone_1 in PLZ_to_NPVM[PLZ]:
            for zone_2 in PLZ_to_NPVM[PLZ]:
                same_PLZ_matrix[zone_mapping[zone_1], zone_mapping[zone_2]] = True
    same_PLZ_matrix = same_PLZ_matrix[0:n_zones, 0:n_zones]

    centroids = dict(
        (row['zone_id'], (row['x_coord'], row['y_coord']))
        for row in pd.read_csv(path_centroids, sep=';').to_dict('records'))

    ##
    logger.info('\tExecuting vehicle generation model...')

    vehicles_by_branch = calc_vehicle_generation(
        path_params_vehicle_generation,
        dim_branch, zone_stats, zone_ids, sep)

    logger.info('\tAggregating branches into segments...')
    # Conversion to segments (branch aggregates)
    vehicles_by_seg = pd.DataFrame(0, index=zone_ids, columns=segment_ids, dtype=np.float32)
    for segment in range(n_segment):
        branches = dim_segment[segment]['branch']
        for branch in branches:
            vehicles_by_seg[segment_ids[segment]] += vehicles_by_branch[branch]

    logger.info('\tExecuting active vehicle model...')

    # Apply proportion of active vehicles to the number of vehicles.
    params_share_active = pd.read_csv(path_params_share_active, sep=',', index_col=0)
    active_vehicles = vehicles_by_seg.copy()
    for segment in active_vehicles.columns:
        if weekday:
            active_vehicles.loc[:, segment] *= params_share_active.at[segment, 'p_active (Mo-Fr)']
        else:
            active_vehicles.loc[:, segment] *= params_share_active.at[segment, 'p_active (Mo-Su)']

    logger.info('\tExecuting number of tours model...')
    params_n_tours = pd.read_csv(path_params_n_tours, sep=',', index_col=0)
    n_tours = active_vehicles.copy()
    for segment in active_vehicles.columns:
        n_tours.loc[:, segment] *= params_n_tours.at[segment, 'nb_tours']

    logger.info('\tConstructing tours...')
    n_tours = n_tours.to_numpy()
    seed = get_setting(config, 'seed', is_allowed_to_be_none=True)
    seed = int(seed) if seed not in [None, ''] else np.random.randint(1000000)
    np.random.seed(seed)

    seeds_by_origin_and_segment = np.random.randint(1, 1000000, size=(n_zones, n_segment))

    logger.info(f'\t\tRandom seed used for tour construction: {seed}')

    params_end_tour: List[Dict[str, Any]] = pd.read_csv(path_params_end_tour, sep=sep,
                                                        index_col=0)  # .to_dict('records')
    params_next_stop: List[Dict[str, Any]] = pd.read_csv(path_params_next_stop, sep=sep,
                                                         index_col=0)  # .to_dict('records')

    for segment in segment_ids:
        logger.debug(f"\t\tSettings segment {segment}:")
        logger.debug(f"\t\t\tprob_return: {params_end_tour.loc[segment, 'prob_return']}")
        logger.debug(f"\t\t\tcost_per_hour: {params_next_stop.loc[segment, 'cost_per_hour']}")
        logger.debug(f"\t\t\tcost_per_km: {params_next_stop.loc[segment, 'cost_per_km']}")

    n_cpu = get_n_cpu(get_setting(config, 'n_cpu', int), 16, logger)

    if n_cpu == 1:

        trip_matrix, tours = calc_tour_construction(
            n_tours, segment_ids,
            tt_matrix, dist_matrix, same_PLZ_matrix,
            land_use, population, jobs,
            params_end_tour, params_next_stop,
            granularity, max_tour_duration_minutes, min_n_stops_for_2_opt,
            n_segment, n_zones,
            seeds_by_origin_and_segment, [0, np.arange(n_zones)])

    else:
        logger.info(f'\t\tNumber of cores used: {n_cpu}')

        p = mp.Pool(n_cpu)

        # Spread the tour construction process over parallel cores
        results_tour_construction = p.map(
            functools.partial(
                calc_tour_construction,
                n_tours, segment_ids,
                tt_matrix, dist_matrix, same_PLZ_matrix,
                land_use, population, jobs,
                params_end_tour, params_next_stop,
                granularity, max_tour_duration_minutes, min_n_stops_for_2_opt,
                n_segment, n_zones,
                seeds_by_origin_and_segment),
            [[cpu, np.arange(cpu, n_zones, n_cpu)] for cpu in range(n_cpu)])

        # Combine results of parallel cores
        trips: List[List[Any]] = []

        for tmp_trips in results_tour_construction:
            trips = trips + tmp_trips

        del results_tour_construction, tmp_trips

        # Update tour_id values to be unique again
        new_tour_ids: List[int] = []
        tmp_tour_id: int = 0

        for i in range(1, len(trips)):
            if trips[i][1] != trips[i - 1][1]:
                tmp_tour_id += 1
            new_tour_ids.append(tmp_tour_id)

        for i in range(1, len(trips)):
            trips[i][1] = new_tour_ids[i - 1]

        p.close()
        p.join()

    logger.info('\tConvert list of trips to DataFrame...')
    trips_df = pd.DataFrame(
        np.array(trips),
        columns=['tour_orig', 'tour_id', 'leg_rank', 'orig_zone', 'dest_zone', 'segment'])
    trips_df['weight'] = granularity

    logger.info('\tBuilding trip-matrix from trips...')
    trip_matrix: np.ndarray = np.zeros((n_segment, n_zones, n_zones), dtype=int)
    np.add.at(trip_matrix, (trips_df['segment'].replace(to_replace=segment_inds).tolist(), trips_df['orig_zone'].astype(int).tolist(),
                              trips_df['dest_zone'].astype(int).tolist()), 1)
    trip_matrix = trip_matrix.astype(np.float32) * granularity

    logger.info('\tCorrection based on daily vehicle distance per branch...')

    if weekday:
        path_daily_dist_per_branch_empirical = path_daily_dist_per_branch_empirical_weekday
    else:
        path_daily_dist_per_branch_empirical = path_daily_dist_per_branch_empirical_full_week

    trip_matrix, trips_df = calc_correction(
        trip_matrix, trips_df, dist_matrix,
        active_vehicles, segment_ids,
        path_daily_dist_per_branch_empirical,
        path_out_daily_dist_per_branch)

    logger.info('\tSumming trip matrix over segments...')
    trip_matrix = np.sum(trip_matrix, axis=0)

    if write_shp or write_csv:
        trips_df['orig_zone'] = [inv_zone_mapping[int(orig_zone)] for orig_zone in trips_df['orig_zone'].values]
        trips_df['dest_zone'] = [inv_zone_mapping[int(dest_zone)] for dest_zone in trips_df['dest_zone'].values]

    if write_shp:
        logger.info('\tWriting shapefile of trips...')
        write_trips_to_shp(path_out_trips_shape, trips_df, centroids)

    if write_csv:
        logger.info('\tWriting trips to .csv...')
        trips_df.to_csv(path_out_trips_csv, sep=sep, index=False)

    logger.info('Creating trip matrix...')

    # Adds to transposed trip matrix in order to maintain matrix symmetry
    trip_matrix = 0.5 * trip_matrix + 0.5 * np.transpose(trip_matrix)

    #####
    if run_parcel_module:
        # Add the parcel trips to the trip_matrix
        for row in parcel_schedules.to_dict('records'):
            trip_matrix[zone_mapping[row['orig_zone']], zone_mapping[row['dest_zone']]] += 1.0

    # Augment the size of the trip matrix to include zones in Liechtenstein as well as Büsingen and Campione d'Italia
    # (OD pairs filled with zeros)
    trip_matrix_incl_external = np.zeros((n_zones + n_external_zones, n_zones + n_external_zones), dtype=np.float32)
    trip_matrix_incl_external[:n_zones, :n_zones] = trip_matrix

    # Write the trip matrix in the OMX format, compatible with VISUM
    if write_omx:
        logger.info('Writing trip matrix to .omx...')

        myfile = omx.open_file(path_out_trip_matrix_omx, 'w')
        myfile['Trips'] = trip_matrix_incl_external
        myfile.create_mapping('NO', list(zone_mapping.keys()))
        myfile.close()

    # Write the trip matrix in the CSV format
    if write_csv:

        logger.info('Writing trip matrix to .csv...')

        with open(path_out_trip_matrix_csv, 'w') as f:
            f.write(f'orig{sep}dest{sep}n_trips\n')
            for orig in range(n_zones):
                for dest in np.where(trip_matrix_incl_external[orig, :] > 0)[0]:
                    n_trips = trip_matrix_incl_external[orig, dest]
                    f.write(f'{orig}{sep}{dest}{sep}{n_trips}\n')


@njit
def draw_choice(
        probs: np.ndarray
) -> int:
    '''
    Draws one choice from an array of probabilities using Monte Carlo Simulation.

    Returns:
        - The chosen alternative (int)
    '''
    n_alt = len(probs)

    cum_prob: float = 0.0
    rand: float = np.random.rand()

    for alt in range(n_alt):
        cum_prob += probs[alt]
        if cum_prob >= rand:
            return alt

    raise Exception(
        '\nError in function "draw_choice", random draw was ' +
        'outside range of cumulative probability distribution.')



def calc_vehicle_generation(
        path_params_vehicle_generation: str,
        dim_branch: List[str],
        zone_stats: pd.DataFrame,
        zone_ids: np.ndarray,
        sep: str
) -> pd.DataFrame:
    '''
    Vehicle Generation model.
        Input = number of jobs (FTE) per branch + population, both at zone level (National Passenger Transport Model 2017)
        Parameters : vehicle rates per branch and for the population
        Output = Nx20 array of number of vehicles pro zone and branch (float)

    Returns:
        - vehicles (pd.DataFrame)
    '''
    # Read parameters
    params_vehicle_generation = pd.read_csv(path_params_vehicle_generation, sep=sep)

    # Initialize the output
    vehicles = pd.DataFrame(0, index=zone_ids, columns=dim_branch)

    # Apply the model, i.e. multiply the vehicle rates by the corresponding explanatory variable
    # for each branch (and each zone).
    for branch in dim_branch:
        vehicles[branch] = (
                zone_stats[branch] *
                params_vehicle_generation.at[0, branch])

    return vehicles


def calc_tour_construction(
        n_tours_by_origin_and_seg: np.ndarray,
        segment_ids: List[str],
        tt_matrix: np.ndarray,
        dist_matrix: np.ndarray,
        same_PLZ_matrix: np.ndarray,
        land_use: np.ndarray,
        population: np.ndarray,
        jobs: np.ndarray,
        params_end_tour: pd.DataFrame,  # List[Dict[str, Any]],
        params_next_stop: pd.DataFrame,  # List[Dict[str, Any]],
        granularity: float,
        max_tour_duration_minutes: float,
        min_n_stops_for_2_opt: int,
        n_segment: int,
        n_zones: int,
        seeds_by_origin_and_segment: np.ndarray,
        cpu_specific_args: Tuple[int, List[int]]
) -> Tuple[np.ndarray, List[List[Any]]]:
    '''
    Construct the tours, based on the Next Stop Location model and the End Tour model.

    Returns:
        - trips (List[List[Any]])
    '''
    cpu = cpu_specific_args[0]
    subset_origin_zones = cpu_specific_args[1]

    # Aggregate total number of tours per segment
    n_tours_by_seg: np.ndarray = np.sum(n_tours_by_origin_and_seg, axis=0)

    # Initialize list of trips
    trips: List[List[Any]] = []

    tour_id: int = 0

    time_start = time.time()  # This is just to measure computational time.

    # To have an idea of the total computational time (approximately linear to the number of tours simulated)
    n_tours_to_simulate_total = np.sum(n_tours_by_origin_and_seg[subset_origin_zones]) / granularity
    n_tours_simulated_total = 0
    next_displayed_threshold = 1 / 100 * n_tours_to_simulate_total

    # Tour building:
    for ind_seg in range(n_segment):  # for each segment
        segment = segment_ids[ind_seg]
        if cpu == 0:
            print(f"Segment: {segment}")

        if n_tours_by_seg[ind_seg] == 0:
            if cpu == 0:
                print(f"\tNo tours in segment {segment}")
            continue

        # Pre-compute destination component for NextStopLocation
        u_next_stop_dest_component: np.ndarray = (
                params_next_stop.loc[segment, 'b_LowDen'] * (land_use == "L") +
                params_next_stop.loc[segment, 'b_Res'] * (land_use == "R") +
                params_next_stop.loc[segment, 'b_Inter'] * (land_use == "I") +
                params_next_stop.loc[segment, 'b_EmpNode'] * (land_use == "E") +
                np.log(
                    (1 + population) +
                    params_next_stop.loc[segment, 'b_jobs_pop'] * (1 + jobs)))

        # The probability of making a return trip at the end of the tour
        prob_return: float = params_end_tour.loc[segment, 'prob_return']

        # The cost figures
        cost_per_hour: float = params_next_stop.loc[segment, 'cost_per_hour']
        cost_per_km: float = params_next_stop.loc[segment, 'cost_per_km']

        cost_matrix: np.ndarray = (
                                          cost_per_hour * tt_matrix / 60 +
                                          cost_per_km * dist_matrix) / 100  # the division by 100 means that the cost is expressed in 100 CHF.
        # This is for numerical reasons more convenient.

        for o in subset_origin_zones:  # iterates over origins
            # Fix the seed
            np.random.seed(seeds_by_origin_and_segment[o, ind_seg])

            # Prints progress (only checks progress once per origin zone, to limit overhead)
            if n_tours_simulated_total > next_displayed_threshold:
                if cpu == 0:
                    print(
                        "\tprogress : {prog:.0%}, elapsed time = {x:.2f} s".format(
                            prog=n_tours_simulated_total / n_tours_to_simulate_total,
                            x=time.time() - time_start))
                next_displayed_threshold += 1 / 100 * n_tours_to_simulate_total

            # Stochastically generate an integer number of tours, whose expected value is
            # n_tours_by_origin_and_seg[o, ind_seg] / granularity
            n_tours_current_origin_expected: float = n_tours_by_origin_and_seg[
                                                         o, ind_seg] / granularity

            # Add 1.0 with a probability equal to the decimal part
            if (n_tours_current_origin_expected % 1) > np.random.rand():
                n_tours_current_origin_expected += 1.0

            # Truncate expected value to an integer number of tours
            n_tours_current_origin = int(np.floor(n_tours_current_origin_expected))

            # Stochastic part (iteration over agents) starts here:
            for i in range(n_tours_current_origin):
                curr_zone: int = o  # current zone
                leg_rank: int = 0  # leg rank in the tour
                tour_duration: float = 0.0
                end_tour: bool = False

                tour_sequence_init: List[int] = [curr_zone]

                while not end_tour:

                    ######## Next Stop Location
                    # First compute the utility of each zone
                    if leg_rank == 0:
                        u_next_stop: np.ndarray = u_next_stop_dest_component + (1-same_PLZ_matrix[
                                                                                                curr_zone, :])*(
                                (params_next_stop.loc[segment, 'b_cost_0'] + params_next_stop.loc[segment,
                                'b_cost_first']) *
                                cost_matrix[curr_zone, :]) + \
                                                  params_next_stop.loc[segment, 'b_same_ZIP'] * same_PLZ_matrix[
                                                                                                curr_zone, :]
                    else:
                        u_next_stop = u_next_stop_dest_component + (1-same_PLZ_matrix[curr_zone, :])*(
                                params_next_stop.loc[segment, 'b_cost_0'] * cost_matrix[curr_zone, :])

                    u_next_stop += (1-same_PLZ_matrix[curr_zone, :])*(
                            params_next_stop.loc[segment, 'b_cost_50'] * np.maximum(
                        cost_matrix[curr_zone, :] - 50 / 100, np.zeros(n_zones, dtype=np.float32)))

                    exp_u_next_stop: np.ndarray = np.exp(u_next_stop)

                    if leg_rank > 0:  # exclude the base as possible destination by giving a proba equal to 0
                        exp_u_next_stop[o] = 0

                    # Converts utility into a probability to be chosen with a simple Multinomial logit.
                    prob_next_stop: np.ndarray = exp_u_next_stop / np.sum(exp_u_next_stop)

                    # Sample the decision (=stochastic part)
                    next_zone = draw_choice(prob_next_stop)

                    # Add the chosen destination to the tour sequence
                    tour_sequence_init.append(next_zone)

                    # Update tour duration
                    tour_duration += tt_matrix[curr_zone, next_zone]

                    # Add the chosen trip to the list of trips, with relevant attributes
                    trips.append([o, tour_id, leg_rank, curr_zone, next_zone, segment])

                    # Updates the current zone and leg rank
                    curr_zone = next_zone
                    leg_rank += 1

                    ######## End Tour
                    if same_PLZ_matrix[curr_zone, o]:
                        end_tour = True
                    else:
                        # Add the missing terms to the utility to continue
                        u_continue: float = (
                                params_end_tour.loc[segment, 'ASC'] +
                                params_end_tour.loc[segment, 'b_cost_return'] * cost_matrix[curr_zone, o] +
                                params_end_tour.loc[segment, 'b_lnStops'] * np.log(leg_rank + 1))
                        if leg_rank == 1:
                            u_continue += params_end_tour.loc[segment, 'cons_2stops']

                        # Samples the decision to continue (=stochastic part)
                        end_tour = np.random.rand() < (1 / (1 + np.exp(u_continue)))

                    # Override End Tour decision if tour lasts longer than 'max_tour_duration'
                    if not end_tour:
                        end_tour = (tour_duration + tt_matrix[curr_zone, o]) >= max_tour_duration_minutes

                    if end_tour:
                        # Samples the decision to make a return trip (=stochastic part)
                        # make_return_trip: bool = (np.random.rand() <= prob_return)
                        if same_PLZ_matrix[curr_zone, o]:
                            make_return_trip = False
                        else:
                            if leg_rank == 1:
                                make_return_trip: bool = (np.random.rand() <= 0.771)
                            else:
                                make_return_trip: bool = (np.random.rand() <= 0.868)
                        if make_return_trip:
                            # Adds the return trip to the list of trips and to the tour sequence
                            tour_sequence_init.append(o)
                            trips.append([o, tour_id, leg_rank, curr_zone, o, segment])

                n_tours_simulated_total += 1

                # Improve the tour sequence if it has many stops
                n_stops = leg_rank + 1 - int(not make_return_trip)

                if n_stops >= min_n_stops_for_2_opt:

                    tour_sequence_post = improve_tour_sequence(
                        np.array(tour_sequence_init, dtype=int), dist_matrix)

                    trips = trips[:-n_stops]

                    for leg_rank in range(n_stops):
                        trips.append([
                            o, tour_id, leg_rank,
                            tour_sequence_post[leg_rank], tour_sequence_post[leg_rank + 1],
                            segment])

                tour_id += 1

    time_now = time.time()

    if cpu == 0:
        print("Time needed for tour construction: {x:.2f} s".format(x=time_now - time_start))

    return trips


def calc_correction(
        trip_matrix_b: np.ndarray,
        trips_df: pd.DataFrame,
        dist_matrix: np.ndarray,
        active_vehicles: pd.DataFrame,
        segment_ids: List[str],
        path_daily_dist_per_branch_empirical: str,
        path_out_daily_dist_per_branch: str
) -> Tuple[np.ndarray, pd.DataFrame]:
    '''
    Correction model. Writes daily distance traveled and ratios in csv.

    Returns:
        - trip matrix (np.ndarray)
        - trips (List[List[Any]])
    '''
    # Compute correction factors (based on average daily distance traveled per branch)
    vkt_per_segment = pd.DataFrame(data=np.sum(trip_matrix_b * dist_matrix, axis=(1, 2)), index=segment_ids)
    veh_per_segment = active_vehicles.sum(axis=0)
    daily_dist_per_segment = (vkt_per_segment[0] / veh_per_segment).to_frame(name='Daily distance (before correction)')
    daily_dist_per_segment['Daily distance (empirical)'] = pd.read_csv(path_daily_dist_per_branch_empirical, sep=',',
                                                                       index_col=0)
    daily_dist_per_segment['Correction factor'] = daily_dist_per_segment['Daily distance (empirical)'] / \
                                                  daily_dist_per_segment['Daily distance (before correction)']
    daily_dist_per_segment.to_csv(path_out_daily_dist_per_branch)

    # update trip matrix and trips
    for ind_seg in range(0, len(segment_ids)):
        trip_matrix_b[ind_seg, :, :] = trip_matrix_b[ind_seg, :, :] * daily_dist_per_segment.loc[
            segment_ids[ind_seg], 'Correction factor']
        trips_df.loc[(trips_df['segment'] == segment_ids[ind_seg]), 'weight'] = trips_df.loc[
                                                                                    trips_df['segment'] == segment_ids[
                                                                                        ind_seg], 'weight'] * \
                                                                                daily_dist_per_segment.loc[
                                                                                    segment_ids[
                                                                                        ind_seg], 'Correction factor']

    return trip_matrix_b, trips_df


def write_trips_to_shp(
        path_out_trips_shape: str,
        trips_shp: pd.DataFrame,
        centroids: Dict[int, Tuple[int, int]]
):
    """
    Writes the DataFrame 'trips_shp' to the specified path in 'path_out_trips_shape'.
    """
    where_tour_id: Dict[int, List[int]] = {}
    col_tour_id = np.where([col == 'tour_id' for col in trips_shp.columns])[0][0]
    for i, row in enumerate(np.array(trips_shp)):
        tour_id = row[col_tour_id]
        try:
            where_tour_id[tour_id].append(i)
        except KeyError:
            where_tour_id[tour_id] = [i]

    trips_shp['n_trips'] = [len(where_tour_id[tour_id]) for tour_id in trips_shp['tour_id'].values]

    trips_shp['x_orig'] = [centroids[orig_zone][0] for orig_zone in trips_shp['orig_zone'].values]
    trips_shp['y_orig'] = [centroids[orig_zone][1] for orig_zone in trips_shp['orig_zone'].values]
    trips_shp['x_dest'] = [centroids[dest_zone][0] for dest_zone in trips_shp['dest_zone'].values]
    trips_shp['y_dest'] = [centroids[dest_zone][1] for dest_zone in trips_shp['dest_zone'].values]

    columns_coords = ['x_orig', 'y_orig', 'x_dest', 'y_dest']

    write_shape_line(path_out_trips_shape, trips_shp, columns_coords)


if __name__ == '__main__':

    config = read_yaml('config.yaml')

    logger, log_stream_handler, log_file_handler = get_logger(config)

    log_and_check_config(config, logger)

    logger.info('')
    logger.info('Run started')

    try:
        mp.freeze_support()
        main(config, logger)
        logger.info('Run finished successfully')
    except Exception:
        logger.exception('Run failed')

    logger.removeHandler(log_stream_handler)
    logger.removeHandler(log_file_handler)

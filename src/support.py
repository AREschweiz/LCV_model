import logging
import logging.handlers
import multiprocessing as mp
import numpy as np
import openmatrix as omx
import os
import pandas as pd
import shapefile as shp
import sys
import time
import yaml

from datetime import datetime
from numba import njit
from typing import Any, Dict, List, Tuple, Type


def get_logger(
        config: Dict[str, Dict[str, Any]]
) -> Tuple[logging.Logger, logging.StreamHandler, logging.handlers.TimedRotatingFileHandler]:
    """
    Create a logger object which prints messages to the terminal and writes a log file.
    """
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)8s]: %(message)s")

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    logger.addHandler(log_stream_handler)

    def float_to_time_stamp(value: float):
        time_stamp = datetime.fromtimestamp(value)
        return (
                str(time_stamp.year).rjust(4, '0') +
                str(time_stamp.month).rjust(2, '0') +
                str(time_stamp.day).rjust(2, '0') +
                '_' +
                str(time_stamp.hour).rjust(2, '0') +
                str(time_stamp.minute).rjust(2, '0') +
                str(time_stamp.second).rjust(2, '0'))

    if config['paths']['outputs'].get('log_folder') is None:
        raise Exception("No path was specified for 'log_folder' under 'paths/outputs' in the YAML file.")
    else:
        path_output = os.path.dirname(os.getcwd()) + config['paths']['outputs'].get('log_folder')

    if not os.path.exists(path_output):
        logger.removeHandler(log_stream_handler)
        raise Exception(
            "Tried to open a log file in the following folder that could not be located: " +
            f"'{os.getcwd()}{config['paths']['outputs']['log_folder']}'.")

    out_file_name = (
            path_output +
            'log_' + float_to_time_stamp(time.time()) + '.txt')

    log_file_handler = logging.handlers.TimedRotatingFileHandler(
        out_file_name, when='D', interval=3, backupCount=10)
    log_file_handler.setFormatter(log_formatter)
    logger.addHandler(log_file_handler)

    return logger, log_stream_handler, log_file_handler


def read_yaml(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Return the contents of the YAML configuration file.
    """
    config = None

    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf8') as yaml_file:
            config = yaml.load(yaml_file, yaml.FullLoader)
    else:
        raise Exception(f"Could not find '{path}'.")

    if not isinstance(config, dict):
        config = {}

    for main_section in ['settings', 'paths', 'expected_fields', 'dimensions']:
        if config.get(main_section) is None:
            raise Exception(f"Could not find main section '{main_section}' in 'config.yaml'.")

    for sub_section in ['inputs', 'outputs']:
        for main_section in ['paths']:
            if config[main_section].get(sub_section) is None:
                raise Exception(
                    f"Could not find sub section '{sub_section}' " +
                    f"under main section '{main_section}' in 'config.yaml'.")

    for sub_section in ['inputs', 'outputs']:
        for key, value in config['paths'][sub_section].items():
            if '${' in value and '}' in value:
                referenced_key = value.split('${')[-1].split('}')[0]
                if referenced_key in config['paths'][sub_section].keys():
                    value = os.path.join(
                        config['paths'][sub_section][referenced_key],
                        value.split('${' + referenced_key + '}')[-1])
                    config['paths'][sub_section][key] = value

    return config


def log_and_check_config(
        config: Dict[str, Dict[str, Any]],
        logger: logging.Logger
):
    """
    Writes the contents of the config file to the logger, checks if the input files exist, and checks if the
    expected headers can be found in the input files.
    """
    logger.info("Settings:")

    for setting_name, setting_value in config['settings'].items():
        setting_value = f"'{setting_value}'" if type(setting_value) == str else setting_value
        logger.info(f"\tsettings/{setting_name}: {setting_value}")

    for input_name, input_path in config['paths']['inputs'].items():
        logger.info(f"\tpaths/inputs/{input_name}: '{input_path}'")

    for output_name, output_path in config['paths']['outputs'].items():
        logger.info(f"\tpaths/outputs/{output_name}: '{output_path}'")

    directory = os.path.dirname(os.getcwd())
    for input_name, input_path in config['paths']['inputs'].items():

        # Check if the input file exists
        if not os.path.exists(directory + input_path):
            logger.warning(f"The following file or folder for '{input_name}' could not be found: '{input_path}'.")

        # Check if all expected headers are found in the input file
        else:
            expected_fields = config['expected_fields'].get(input_name)
            sep = config['settings'].get('sep')

            if type(expected_fields) == list and sep is not None:
                with open(directory + input_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().replace('\n', '').split(config['settings'].get('sep'))
                for expected_field in expected_fields:
                    if expected_field not in first_line:
                        logger.warning(f"Field '{expected_field}' was not found in input file '{input_name}'.")
                        logger.debug(f"\t(path: '{input_path}')")
                        logger.debug(f"\t(headers as read from file: {first_line})")


def get_dimension(
        config: Dict[str, Dict[str, Any]],
        name: str,
        is_allowed_to_be_none: bool = False
) -> List[Dict[str, Any]]:
    """
    Gets the contents of a dimension specified in the YAML file.
    """
    dimension = config['dimensions'].get(name)

    if dimension is None and not is_allowed_to_be_none:
        raise Exception(f"Could not find dimension '{name}' under main section 'dimensions' in 'config.yaml'.")

    return dimension


def get_setting(
        config: Dict[str, Dict[str, Any]],
        name: str,
        required_type: Type = None,
        is_allowed_to_be_none: bool = False
) -> Any:
    """
    Gets the value of a setting specified in the YAML file.
    """
    value = config['settings'].get(name)

    if value is None and not is_allowed_to_be_none:
        raise Exception(f"Could not find setting '{name}' under main section 'settings' in 'config.yaml'.")

    if required_type is not None:

        if type(value) == int and required_type in [int, float]:
            return value

        if type(value) != required_type:
            raise Exception(
                f"A value ('{value}') of type {type(value)} instead of required type {required_type} " +
                f"was specified for '{name}' under main section 'settings' in 'config.yaml'.")

    return value


def get_input_path(
        config: Dict[str, Dict[str, Any]],
        name: str,
        is_allowed_to_be_none: bool = False
) -> str:
    """
    Gets the path of a specified input file in the YAML file.
    """
    parent_directory = os.path.dirname(os.getcwd())
    path = parent_directory + config['paths']['inputs'].get(name)

    if path is None and not is_allowed_to_be_none:
        raise Exception(f"Could not find '{name}' under 'inputs' under main section 'paths' in the config file (yaml).")

    return path


def get_output_path(
        config: Dict[str, Dict[str, Any]],
        name: str,
        is_allowed_to_be_none: bool = False
) -> str:
    """
    Gets the path of a specified output file in the YAML file.
    """
    parent_directory = os.path.dirname(os.getcwd())
    path = parent_directory + config['paths']['outputs'].get(name)

    if path is None and not is_allowed_to_be_none:
        raise Exception(f"Could not find '{name}' under 'outputs' under main section 'paths' in 'config.yaml'.")

    return path


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

    n_records_update = int(n_records / 10)

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

        if i % n_records_update == 0:
            print(f'\t{round((i / n_records) * 100, 1)}%')

    w.close()

    print('\t100.0%')


def get_omx_matrix(
        path_omx_matrix: str,
        n_zones: int,
        n_external_zones: int,
        sub_table_name: str = ''
) -> Tuple[np.ndarray, Dict[int, int], np.ndarray]:
    '''
    Get a matrix saved in OMX format.

    Returns:
        - matrix (np.ndarray)
        - zone_mapping (Dict[int, int])
        - zone_ids (np.ndarray)
    '''
    # Import matrix
    with omx.open_file(path_omx_matrix, 'r') as omx_file:
        sub_table_name = omx_file.list_matrices()[0] if sub_table_name == '' else sub_table_name
        matrix = np.array(omx_file[sub_table_name]).astype(np.float32)
        zone_mapping = omx_file.mapping('NO')

    if n_zones + n_external_zones != matrix.shape[0]:
        raise Exception(
            f"The OMX file contains {matrix.shape[0]} rows, " +
            f"but {n_zones + n_external_zones} rows were expected. ({path_omx_matrix})")

    # remove the  11 zones in Liechtenstein and 2 enclaves, which occupy the last indices.
    matrix = matrix[:n_zones, :n_zones]
    zone_ids = np.array(list(zone_mapping.keys()))
    zone_ids = zone_ids[:n_zones]

    return (matrix, zone_mapping, zone_ids)


@njit
def improve_tour_sequence(
        sequence_init: np.ndarray,
        dist_matrix: np.ndarray
):
    """
    Improve a sequence of visiting zones by swapping its order and accepting swaps if they reduce the total distance.
    """
    n_stops = len(sequence_init)

    current_tour_distance = 0.0
    for i in range(n_stops - 1):
        current_tour_distance += dist_matrix[sequence_init[i], sequence_init[i + 1]]

    sequence_post = sequence_init.copy()

    for shift_loc_a in range(1, n_stops - 1):
        for shift_loc_b in range(1, n_stops - 1):

            if shift_loc_a == shift_loc_b:
                continue

            sequence_swapped = sequence_post.copy()

            tmp_loc_a = sequence_swapped[shift_loc_a]
            tmp_loc_b = sequence_swapped[shift_loc_b]
            sequence_swapped[shift_loc_a] = tmp_loc_b
            sequence_swapped[shift_loc_b] = tmp_loc_a

            swapped_tour_distance = 0.0
            for i in range(n_stops - 1):
                swapped_tour_distance += dist_matrix[sequence_swapped[i], sequence_swapped[i + 1]]

            if swapped_tour_distance < current_tour_distance:
                sequence_post = sequence_swapped.copy()
                current_tour_distance = swapped_tour_distance

    return sequence_post


def get_n_cpu(
        n_cpu: int,
        max_n_cpu: int,
        logger: logging.Logger
) -> int:
    """
    Determine the number of cores over which to spread processes.
    """
    if n_cpu is not None:

        if n_cpu > mp.cpu_count():
            n_cpu = max(1, min(mp.cpu_count() - 1, max_n_cpu))
            logger.warning(
                f"The value of 'n_cpu' is too high. This pc has only {mp.cpu_count()} processors available. " +
                f"Therefore, 'n_cpu' is set internally to {n_cpu}.")

        if n_cpu < 1:
            n_cpu = max(1, min(mp.cpu_count() - 1, max_n_cpu))

    else:
        n_cpu = max(1, min(mp.cpu_count() - 1, max_n_cpu))

    return n_cpu


def get_zone_stats(
        path_zone_stats: str,
        sep: str
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Get the relevant zonal characteristics.

    Returns:
        - zone_stats (pd.DataFrame)
        - population (np.ndarray)
        - jobs (np.ndarray)
        - land_use (np.ndarray)
    '''
    # Import as DataFrame and store the most important variables separately
    zone_stats = pd.read_csv(path_zone_stats, sep=sep, index_col=0)
    n_zones = len(zone_stats.index)

    # Population of each zone
    population = zone_stats['Pop'].to_numpy(dtype=np.float32)

    # Jobs of each zone
    jobs = zone_stats['Jobs'].to_numpy(dtype=np.float32)

    # Land use of each zone
    if 'LandUse' in zone_stats.columns:
        land_use = zone_stats['LandUse'].to_numpy(dtype=str)
    else:
        area = zone_stats['Area'].to_numpy(dtype=np.float32)
        land_use = np.array(["L"] * n_zones, dtype=str)
        for zone in range(n_zones):
            if population[zone] / area[zone] <= 100 and jobs[zone] / area[zone] <= 100:
                land_use[zone] = "L"  # Low-density area
            elif population[zone] / area[zone] > 100 and jobs[zone] <= 2 * population[zone]:
                land_use[zone] = "R"  # Residential area
            elif jobs[zone] / area[zone] <= 3000:
                land_use[zone] = "I"  # Industrial area
            else:
                land_use[zone] = "E"  # Employment node

    return (zone_stats, population, jobs, land_use)


def furness(
        matrix: np.ndarray,
        cons_sum_rows: np.ndarray,
        cons_sum_cols: np.ndarray,
        epsilon_criterion: float,
        max_steps: int) -> np.ndarray:
    """
  matrix: matrix to be balanced
  cons_sum_rows : vector of constraints for the sum of rows
  cons_sum_cols : vector of constraints for the sum of columns
  EpsilonCriterion: required precision (stopping criterion)
  max_steps: maximum number of steps (stopping criterion)
  """
    cons_sum_rows = np.atleast_1d(
        np.squeeze(cons_sum_rows))  # to ensure that the constraints are always 1-dimensional arrays
    cons_sum_cols = np.atleast_1d(np.squeeze(cons_sum_cols))
    matrix = np.atleast_2d(matrix)

    if np.sum(cons_sum_rows) * np.sum(cons_sum_cols) > 0:
        if np.sum(matrix) == 0:
            print("the initial matrix contains only zeros, but not the constraints")
        else:
            step_nb: int = 0
            sum_rows = np.sum(matrix, 0)
            sum_cols = np.sum(matrix, 1)
            num_rows = np.shape(matrix)[0]
            num_cols = np.shape(matrix)[1]
            epsilon = epsilon_criterion + 1  # simply to ensure that we enter the loop.
            a = np.ones(num_rows, dtype=np.float32)
            b = np.ones(num_cols, dtype=np.float32)
            if max(np.logical_and(sum_rows == 0, cons_sum_rows > 0)):
                print('Problem not solvable because of row constraints')
            if max(np.logical_and(sum_cols == 0, cons_sum_cols > 0)):
                print('Problem not solvable because of column constraints')
            while step_nb <= max_steps and epsilon > epsilon_criterion:
                epsilon = 0
                step_nb += 1
                a_old = a
                b_old = b
                a = np.divide(cons_sum_cols, np.matmul(matrix, b), out=np.ones_like(cons_sum_cols),
                              where=cons_sum_cols != 0)
                b = np.divide(cons_sum_rows, np.matmul(a, matrix), out=np.ones_like(cons_sum_rows),
                              where=cons_sum_rows != 0)
                epsilon = np.max([np.max(abs(np.divide(a, a_old) - 1)), np.max(abs(np.divide(b, b_old) - 1))])
            print("epsilon:{0}".format(epsilon))
    return a.reshape((num_rows, 1)) * matrix * b


def get_and_write_same_zip_matrix(mapping_npvm_zip: dict, n_zones: int, npvm_mapping: dict, output_path: str) -> np.ndarray:
    # returns a binary matrix for the zones npvm, with 1 iff the zones in row and column have the same zip code
    # mapping_npvm_zip: is a dict with zone npvm as key and zip as value
    # npvm_mapping: is a dict with zone npvm as key and index in the npvm matrices as value
    zip_set = set(list(mapping_npvm_zip.values()))

    # construct a dict where key = zip and value = list of npvm zones corresponding to this PLZ
    zip_to_npvm = {zip_code: [] for zip_code in zip_set}
    for zone_npvm, zip_code in mapping_npvm_zip.items():
        zip_to_npvm[zip_code].append(zone_npvm)

    # Fill in same_zip_matrix
    same_zip_matrix = np.zeros(shape=[n_zones, n_zones], dtype=bool)
    for zip_code in zip_set:
        for zone_1 in zip_to_npvm[zip_code]:
            for zone_2 in zip_to_npvm[zip_code]:
                same_zip_matrix[npvm_mapping[zone_1], npvm_mapping[zone_2]] = True

    npvm_zones_internal = [key for key in npvm_mapping.keys() if int(key) < 700101001]
    # Save it as omx
    myfile = omx.open_file(output_path, 'w')
    myfile['same_zip'] = same_zip_matrix
    myfile.create_mapping('NO', npvm_zones_internal)
    myfile.close()

    return same_zip_matrix

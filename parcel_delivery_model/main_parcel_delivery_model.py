# Main script for the simulation of LCV-model (agent-based version)
# Authors:
# - Sebastiaan Thoen, Significance BV
# - Raphael Ancel, Swiss Federal Office for Spatial Development

import logging
from typing import Any, Dict, List, Tuple
from src.support import (read_yaml, get_logger, log_and_check_config)
import parcel_demand_synthesis
import parcel_delivery_scheduling

def main(config: Dict[str, Dict[str, Any]], logger: logging.Logger):

    logger.info('Parcel Demand Synthesis module...')

    parcel_demand = parcel_demand_synthesis.main(config)

    logger.info('Parcel Delivery Scheduling module...')

    parcel_schedules = parcel_delivery_scheduling.main(config, parcel_demand)

if __name__ == '__main__':

    config = read_yaml('config_parcel.yaml')

    logger, log_stream_handler, log_file_handler = get_logger(config)

    log_and_check_config(config, logger)

    logger.info('')
    logger.info('Run started')

    try:
        main(config, logger)
        logger.info('Run finished successfully')
    except Exception:
        logger.exception('Run failed')

    logger.removeHandler(log_stream_handler)
    logger.removeHandler(log_file_handler)

###################
# This script generates an histogram comparing the trip length distribution in the LCV survey with the one from the simulation.
# To be fair, the comparison should be done for the same year (2013)
###################

import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path
folder_project = Path.cwd().parent
from src.support import (
    read_yaml, get_setting, get_input_path, get_omx_matrix, get_output_path)
import matplotlib.pyplot as plt

# graphic parameters
plt.rcParams.update({
    "text.usetex": True
})
label_size = 14

# read config file
config = read_yaml(str(folder_project / 'src' / 'config.yaml'))
sep = get_setting(config, 'sep', str)
n_external_zones = get_setting(config, 'n_external_zones', int)
n_zones = 7965

# load trip matrix
trip_matrix = get_omx_matrix(get_output_path(config, 'trip_matrix_omx'), n_zones=n_zones, n_external_zones=n_external_zones)[0]
# load distance matrix
dist_matrix, zone_mapping, zone_ids = get_omx_matrix(get_input_path(config, 'dist_matrix'), n_zones=n_zones, n_external_zones=n_external_zones)
# load travel time matrix
travel_time_matrix = get_omx_matrix(get_input_path(config, 'tt_matrix'), n_zones=n_zones, n_external_zones=n_external_zones)[0]
# load same_zip_matrix
same_zip_matrix = get_omx_matrix(get_input_path(config, 'same_zip_matrix'), n_zones, 0)[0]

trip_matrix_flat = trip_matrix.flatten()
dist_matrix_flat = dist_matrix.flatten()
travel_time_matrix_flat = travel_time_matrix.flatten()
same_zip_matrix_flat = same_zip_matrix.flatten()

external_trips = np.where(1 - same_zip_matrix, trip_matrix, 0)
external_trips_flat = external_trips.flatten()

# Define bin edges
bin_edges = np.arange(start=0, stop=150, step=5)

fig, axs = plt.subplots(1, 2)
sns.histplot(ax=axs[0], x=dist_matrix_flat, weights=external_trips_flat/sum(external_trips_flat), bins=bin_edges, label='Model')
axs[0].set(xlabel='Distance [km]', ylabel='Density', title='Trip length')

sns.histplot(ax=axs[1], x=travel_time_matrix_flat, weights=external_trips_flat/sum(external_trips_flat), bins=bin_edges, label='Model')
axs[1].set(xlabel='Travel time [min]', ylabel='Density', title='Trip duration')

# Load empirical data
trips_survey = pd.read_csv(folder_project / 'data' / 'trips_lcv_data.csv', sep=',')
trips_survey = trips_survey[(trips_survey['DIST'] >= 0) * (trips_survey['TTC'] >= 0)]
trips_survey = trips_survey[(trips_survey['BRANCH'] < 100)]
trips_survey = trips_survey[~((trips_survey['ORIG'] == trips_survey['DEST']) & (4 * trips_survey['DIST'] <= trips_survey['DIST_SURVEY'] ))] # data cleaning
trips_survey_external = trips_survey[~(trips_survey['ORIG'] == trips_survey['DEST'])]

weights = trips_survey_external['STATISTICAL_WEIGHT']/sum(trips_survey_external['STATISTICAL_WEIGHT'])
sns.histplot(ax=axs[0], x=trips_survey_external['DIST'], weights=weights, bins=bin_edges, label='Survey')
sns.histplot(ax=axs[1], x=trips_survey_external['TTC'], weights=weights, bins=bin_edges, label='Survey')
axs[0].legend()
axs[1].legend()
plt.tight_layout()
plt.savefig(folder_project / 'outputs' / 'figures' / 'trip_distributions.svg', bbox_inches='tight', format='svg')

# Compute proportion of trips inside ZIP code
prop_same_zip_simu = sum(trip_matrix_flat*same_zip_matrix_flat)/sum(trip_matrix_flat)
prop_same_zip_empirical = 1 - sum(trips_survey_external['STATISTICAL_WEIGHT'].values)/sum(trips_survey['STATISTICAL_WEIGHT'].values)

print(f'prop_same_zip_simu={str(prop_same_zip_simu)}')
print(f'prop_same_zip_empirical={str(prop_same_zip_empirical)}')


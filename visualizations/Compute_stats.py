###################
# This script computes some statistics about the LCV survey.
###################

import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
plt.rcParams.update({
    "text.usetex": True
})

folder_project = Path.cwd().parent

# Load LCV data
tours = pd.read_csv(folder_project / 'data' / 'tours_lcv_data.csv', sep=',')
trips = pd.read_csv(folder_project / 'data' / 'trips_lcv_data.csv', sep=',')
surveys_type_1 = pd.read_csv(folder_project / 'data' / 'LWE_2013_VEHICLE.csv', sep=";")

# Average daily distance over all branches for surveys of type 1
daily_dist_type_1 = np.average(surveys_type_1['KM_TOTAL'], weights=surveys_type_1['wh_tot_cal'])

# Check whether the recorded distances are in line with the modeled ones
trips_localized = trips[trips['DIST']>=0]
trips_localized = trips_localized[~((trips_localized['ORIG'] == trips_localized['DEST']) &
                                    (2 * 2*trips_localized['DIST'] <= trips_localized['DIST_SURVEY']))]

fig, ax = plt.subplots(1)
sns.scatterplot(ax=ax, data=trips, x='DIST_SURVEY', y='DIST')
ax.set(xlabel='Reported distance [km]', ylabel='Distance according to NPVM2017 [km]')
ax.set_xlim([1, 1000])
ax.set_ylim([1, 1000])
plt.xscale('log')
plt.yscale('log')
plt.savefig(folder_project / 'outputs' / 'figures' / 'dist_survey_vs_NPVM.svg', bbox_inches='tight', format='svg')

VKT_reported = (trips_localized['DIST_SURVEY'] * trips_localized['STATISTICAL_WEIGHT']).sum()
VKT_computed = (trips_localized['DIST'] * trips_localized['STATISTICAL_WEIGHT']).sum()
print(f'Reported distance is in average {100*(VKT_reported/VKT_computed-1)} \% greather than according to the model.')
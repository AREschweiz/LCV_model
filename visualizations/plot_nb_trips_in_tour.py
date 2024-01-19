###################
# This script generates an histogram comparing the distribution of the number of trips in a tour in the LCV survey with the one from the simulation.
# To be fair, the comparison should be done for the same year (2013)
###################

import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True
})

folder_project = Path.cwd().parent

# Load LCV data  (tours)
tours_survey = pd.read_csv(folder_project / 'data' / 'tours_lcv_data.csv', sep=',')
tours_survey = tours_survey.rename(columns={'N_TRIPS': 'number of trips', 'STATISTICAL_WEIGHT': 'weight'})
tours_survey = tours_survey[~((tours_survey['INTERNAL']>1) & (tours_survey['RECORDED_DIST_FIRST_LEG'] > 4 * tours_survey['DIST_RETURN']))]
tours_survey['weight'] = tours_survey['weight']/sum(tours_survey['weight'])

# Load list of individual trips resulting from model
trips_model = pd.read_csv(folder_project / 'outputs' / 'run2013' / 'trips.csv', sep=';')
tours_model = trips_model.groupby(by='tour_id')['weight'].agg(['mean', 'count']).rename(columns={'mean':'weight', 'count': 'number of trips'})
tours_model['weight'] = tours_model['weight']/sum(tours_model['weight'])

fig, ax = plt.subplots(1, 1)
sns.histplot(ax=ax,data=tours_model, x='number of trips', weights='weight', label='Model', discrete=True)
sns.histplot(ax=ax,data=tours_survey, x='number of trips', weights='weight', label='Survey', discrete=True)
ax.set(xlabel='Number of trips in tour', ylabel='Density')

plt.legend()
plt.savefig( folder_project / 'outputs' / 'figures' / 'nb_trips_in_tour.svg', bbox_inches='tight', format='svg')


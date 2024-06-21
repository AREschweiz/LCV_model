###################
# This script generates an histogram comparing the distribution of the number of trips in a tour in the LCV survey with the one from the simulation.
# To be fair, the comparison should be done for the same year (2013)
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

# Load LCV data  (tours)
tours_survey = pd.read_csv(folder_project / 'data' / 'tours_lcv_data.csv', sep=',')
tours_survey = tours_survey.rename(columns={'N_TRIPS': 'number of trips', 'STATISTICAL_WEIGHT': 'weight'})
tours_survey = tours_survey[tours_survey['BRANCH']<100]
tours_survey = tours_survey[~((tours_survey['INTERNAL']>0) & (tours_survey['RECORDED_DIST_FIRST_LEG'] > 4 * tours_survey['DIST_RETURN']))]
tours_survey['weight'] = tours_survey['weight']/sum(tours_survey['weight'])

# Descriptive stats on the number of trips per tour
def weighted_median(values, weights):
    sorted_indices = np.argsort(values)
    sorted_values = np.array(values)[sorted_indices]
    sorted_weights = np.array(weights)[sorted_indices]
    cumulative_weight = np.cumsum(sorted_weights)
    median_idx = np.where(cumulative_weight >= cumulative_weight[-1] / 2)[0][0]
    return sorted_values[median_idx]

# weighted average
weighted_average = (tours_survey['weight'] * tours_survey['number of trips']).sum() / tours_survey['weight'].sum()

# Weighted median
weighted_median_value = weighted_median(tours_survey['number of trips'], tours_survey['weight'])

# Weighted variance
weighted_variance = ((tours_survey['weight'] * (tours_survey['number of trips'] - weighted_average) ** 2).sum()) /\
                    tours_survey['weight'].sum()

print("Weighted Average:", weighted_average)
print("Weighted Median:", weighted_median_value)
print("Weighted Variance:", weighted_variance)

# Load list of individual trips resulting from model
trips_model = pd.read_csv(folder_project / 'outputs' / 'run2013' / 'trips.csv', sep=';')
tours_model = trips_model.groupby(by='tour_id')['weight'].agg(['mean', 'count']).rename(columns={'mean':'weight', 'count': 'number of trips'})
tours_model['weight'] = tours_model['weight']/sum(tours_model['weight'])
p=1/weighted_average
geometric_dist = pd.DataFrame(index=range(1,100), data=np.transpose(range(1, 100)), columns=['number of trips'])
geometric_dist['proba'] = 100* p*(1-p)**(geometric_dist['number of trips']-1)
geometric_dist['cum_proba'] = 100 * (1-p**geometric_dist['number of trips'])

check_mean_geo = (geometric_dist['proba']*geometric_dist['number of trips']).sum() / geometric_dist['proba'].sum()
check_mean_survey = (tours_survey['weight']*tours_survey['number of trips']).sum() / tours_survey['weight'].sum()

fig, ax = plt.subplots(2)
# pdf
sns.histplot(ax=ax[0], data=tours_model, x='number of trips', weights='weight', label='Model', discrete=True, stat='percent')
sns.histplot(ax=ax[0], data=tours_survey, x='number of trips', weights='weight', label='Survey', discrete=True, stat='percent')
sns.scatterplot(ax=ax[0], data=geometric_dist, x='number of trips', y='proba', label='Geometric distribution (p=1/2.24)')
ax[0].set(xlabel='Number of trips in tour', ylabel='Proportion of tours [\%]')
ax[0].set_xlim([0, 19])
ax[0].xaxis.set_major_locator(MultipleLocator(1))
ax[0].legend()
#cdf
sns.histplot(ax=ax[1], data=tours_model, x='number of trips', weights='weight', label='Model', discrete=True, stat='percent', cumulative=True)
sns.histplot(ax=ax[1], data=tours_survey, x='number of trips', weights='weight', label='Survey', discrete=True, stat='percent', cumulative=True)
sns.scatterplot(ax=ax[1], data=geometric_dist, x='number of trips', y='cum_proba', label='Geometric distribution (p=1/2.24)')
ax[1].set(xlabel='Number of trips in tour', ylabel='Cumulative proportion of tours [\%]')
ax[1].set_xlim([0, 19])
ax[1].set_ylim([0, 100])
ax[1].xaxis.set_major_locator(MultipleLocator(1))
ax[1].get_legend().set_visible(False)

plt.savefig(folder_project / 'outputs' / 'figures' / 'nb_trips_in_tour.svg', bbox_inches='tight', format='svg')


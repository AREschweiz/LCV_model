#############
# This script gathers the parameters of the module NextStopLocation from the output files of Apollo in a single file with the appropriate format.
# It also makes a figure illustrating the impact of generalized cost on utility (for the technical report).
#############

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True
})
from pathlib import Path

folder_project = Path.cwd().parent

repository = folder_project/'outputs'/'NextStopLocation'

segment_files = {
    'C': 'next_stop_location_C',
    'F': 'next_stop_location_F',
    'G': 'next_stop_location_G',
    'H (49-52)': 'next_stop_location_H',
    'N': 'next_stop_location_N',
    'Other': 'next_stop_location_other'
}
'''## generate csv to be read in simulation
parameters = pd.DataFrame(columns=segment_files.keys(), index=['b_LowDen', 'b_Res', 'b_Inter', 'b_EmpNode', 'b_same_ZIP',
                                                            'b_cost_first',
                                                               'b_cost_0', 'b_cost_50', 'b_jobs_pop',
                                                          'cost_per_hour', 'cost_per_km'], dtype=np.float64)

for segment, file_name in segment_files.items():
    segment_par = pd.read_csv(repository / (file_name+'_estimates.csv'), sep=',', index_col=0)
    parameters[segment] = segment_par['Estimate']
parameters = parameters.fillna(0)
parameters = parameters.transpose()

parameters.to_csv(folder_project / 'parameters'/ 'NextStopLocation.csv', sep=';')'''

## Make plot
parameters = pd.read_csv(folder_project / 'parameters'/ 'NextStopLocation.csv', sep=';', index_col=0)
data_to_plot = pd.DataFrame(data= np.arange(start=0, stop=100, step=1, dtype=np.float64), columns=['cost']) # tmp: we will add more columns in case b_cost_first >0

for segment in parameters.index:
    data_to_plot[segment] = parameters.loc[segment, 'b_cost_0'] * data_to_plot['cost']/100 + \
                            parameters.loc[segment, 'b_cost_50'] * np.maximum(data_to_plot['cost']-50, 0)/100
    if abs(parameters.loc[segment, 'b_cost_first']) > 0 :
        data_to_plot[segment + ' (intermediary)'] = data_to_plot[segment] # addition of a new column
        data_to_plot = data_to_plot.rename(columns={segment: segment + ' (first)'})
        data_to_plot[segment+' (first)'] = data_to_plot[segment + ' (intermediary)'] + \
                                           parameters.loc[segment, 'b_cost_first'] * data_to_plot['cost']/100

data_to_plot = data_to_plot.set_index(keys='cost')

ax = sns.lineplot(data=data_to_plot)
ax.set(ylabel='Contribution to the utility', xlabel='Generalized travel cost (CHF)')
plt.xlim(0, 100)
plt.savefig(folder_project / 'outputs' / 'figures' / 'marginal_utility_cost.svg', bbox_inches='tight',
            format='svg')
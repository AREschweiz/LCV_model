import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

folder_project = Path.cwd().parent
sns.set()

repository = folder_project/'outputs'/'EndTour'
segment_files = { # These should be in the same order as the segments in the config file (yaml).
    'Private': 'EndTour_Private_considering_open_tours_without_lnStops',
    'C': 'EndTour_C_considering_open_tours_without_2stops_and_cost',
    'F': 'EndTour_F_considering_open_tours_without_lnstops_and_cost',
    'G': 'EndTour_G_considering_open_tours_without_lnstops',
    'H': 'EndTour_H_considering_open_tours_without_lnstops',
    'N': 'EndTour_N_considering_open_tours_tmp',
    'Other': 'EndTour_Other_considering_open_tours_wo_lnStops_return'
}

## generate csv to be read in simulation
parameters = pd.DataFrame(columns=segment_files.keys(), index=['ASC', 'cons_2stops', 'b_lnStops',
                                                               'b_cost_return', 'prob_return'], dtype=np.float64)

for segment, file_name in segment_files.items():
    segment_par = pd.read_csv(repository / (file_name + '_estimates.csv'), sep=',', index_col=0)
    parameters[segment] = segment_par['Estimate']
parameters = parameters.fillna(0)
parameters = parameters.transpose()
parameters['prob_return'] = 0.79879

parameters.to_csv(folder_project / 'parameters' / 'EndTour.csv', sep=';')

a=1
## generate latex script to display table
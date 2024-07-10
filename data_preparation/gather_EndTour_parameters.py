#############
# This script gathers the parameters of the module EndTour from the output files of Apollo in a single file with the appropriate format.
#############
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path

folder_project = Path.cwd().parent
sns.set()

repository = folder_project/'outputs'/'end_tour'
segment_files = { # These should be in the same order as the segments in the config file (yaml).
    'Private': 'EndTour_Private',
    'C': 'EndTour_C',
    'F': 'EndTour_F',
    'G': 'EndTour_G',
    'H (49-52)': 'EndTour_H (49-52)',
    'N': 'EndTour_N',
    'Other': 'EndTour_Other'
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
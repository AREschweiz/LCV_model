###################
# This script generates csv files with parameters for the module "Active Vehicles", "Number of Tours" and "Correction".
###################

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List

folder_project = Path.cwd().parent

#######################################################################
## Prepare some mappings
#######################################################################

# generate dict mapping NOGA codes from LWE to NOGA letter
NOGA_BFS_df = pd.read_excel(folder_project / 'data' / 'NOGA_BFS.xlsx')
NOGA_BFS = dict(zip(NOGA_BFS_df['CODE_LWE'].values.tolist(), NOGA_BFS_df['NOGA08'].values.tolist()))

# generate dict mapping NOGA letter to NOGA group
NOGA_to_group = dict(zip(NOGA_BFS_df['NOGA08'].values.tolist(), NOGA_BFS_df['Group'].values.tolist()))

########################################################################
## Probability that a vehicle is active ---------------------------------
########################################################################

# Load separate file with all LCV surveys, their status and NOGA
survey_status = pd.read_csv(folder_project / 'data' / 'LWE_2013_ARE_final_weights.csv', sep=";")

# Load NOGA mapping (the file from BFS has NOGA code with 6 digits)
NOGA_structure = pd.read_excel(folder_project / 'data' / 'NOGA_full_mapping.xlsx')

# merge NOGA mapping with survey_status to add the code NOGA 1 (letter) and 2
survey_status = pd.merge(left=survey_status, right=NOGA_structure, left_on='NACE_CODE', right_on='level5', how='left')
survey_status['level1'] = survey_status['level1'].fillna('Private')

# drop rows with NOGA 2 equal to 53 (postal and courier activities)
survey_status = survey_status[survey_status['level2'] != 53]

# replace NOGA letter by NOGA group
survey_status = survey_status.replace({'level1': NOGA_to_group})

# Include day of the week
survey_status['SurveyDay'] = round(survey_status['OID']/10000, 0) % 100
survey_days = pd.read_excel(folder_project / 'data' / 'Stichtage_LWE13.xlsx')
survey_days = survey_days[['Stichtag_Nummer', 'DayOfWeek']]
survey_status = pd.merge(left=survey_status, right=survey_days, left_on='SurveyDay', right_on='Stichtag_Nummer')

# group_by NOGA group and compute the proportion of active vehicles (Monday-Sunday)
grouped_NOGA_status = survey_status.groupby(['level1', 'METASTATUS'])['wh_tot_cal'].sum().reset_index()
grouped_NOGA = grouped_NOGA_status.groupby(['level1'])['wh_tot_cal'].transform('sum')
grouped_NOGA_status['Proportion'] = grouped_NOGA_status['wh_tot_cal'] / grouped_NOGA

prop_active_MoSu = pd.pivot(data=grouped_NOGA_status, index=['level1'], columns='METASTATUS', values='Proportion')
prop_active_MoSu = prop_active_MoSu['g'].to_frame(name='p_active')
prop_active_MoSu = prop_active_MoSu.rename(columns={prop_active_MoSu.columns[0]: 'p_active'})
prop_active_MoSu.name = 'p_active'

# repeat the same process for weekdays only (Monday-Friday)
grouped_NOGA_status = survey_status[survey_status['DayOfWeek'].isin(['MO', 'DI', 'MI', 'DO', 'FR'])].groupby(['level1', 'METASTATUS'])['wh_tot_cal'].sum().reset_index()
grouped_NOGA = grouped_NOGA_status.groupby(['level1'])['wh_tot_cal'].transform('sum')
grouped_NOGA_status['Proportion'] = grouped_NOGA_status['wh_tot_cal'] / grouped_NOGA

prop_active_MoFr = pd.pivot(data=grouped_NOGA_status, index=['level1'], columns='METASTATUS', values='Proportion')
prop_active_MoFr = prop_active_MoFr['g'].to_frame(name='p_active')
prop_active_MoFr = prop_active_MoFr.rename(columns={prop_active_MoFr.columns[0]: 'p_active'})
prop_active_MoFr.name = 'p_active'


prop_active = pd.merge(left=prop_active_MoSu, right=prop_active_MoFr, suffixes=(" (Mo-Su)"," (Mo-Fr)"), how='outer', left_index=True, right_index=True)
prop_active = prop_active.fillna(0)
prop_active.to_csv(folder_project / 'parameters' /' prop_active.csv')

########################################################################
##################### Number of tours
########################################################################
tours = pd.read_csv(folder_project / 'data' / 'tours_lcv_data.csv')
# replace numeric branch code from LWE by NOGA letter
tours = tours.replace({'BRANCH': NOGA_BFS}) # from number to letter
tours = tours.replace({'BRANCH': NOGA_to_group}) # from letter to group

nb_tours =tours.groupby(['OID', 'BRANCH', 'STATISTICAL_WEIGHT'], as_index=False).agg(nb_tours=('TOUR_ID',"count"))

weighted_avg = lambda x: np.average(x, weights=nb_tours.loc[x.index, "STATISTICAL_WEIGHT"])

# nb tours by NOGA group
nb_tours_by_branch = nb_tours.groupby(["BRANCH"], as_index=True).agg(nb_tours=("nb_tours", weighted_avg))

# nb OIDs by purpose, curb weight and branch
counts_by_branch = nb_tours.groupby(["BRANCH"], as_index=True).agg(counts=("OID", "count"))

#Generate copy in string format
nb_tours_by_branch_string = nb_tours_by_branch.round(decimals=2).astype(str)

nb_tours_by_branch_string.to_csv(folder_project / 'parameters' / 'nb_tours_string.csv')

# reformat csv version so that it is compatible with main.py
nb_tours_by_branch.to_csv(folder_project / 'parameters' / 'nb_tours.csv')

########################################################################
### Daily distance based on surveys of type 1
########################################################################
# Load list of surveys of type 1 (read original file from BFS)
surveys_type_1 = pd.read_csv(folder_project / 'data' / 'LWE_2013_VEHICLE.csv', sep=";")

# import full NOGA code, to be able to exclude post & mail
surveys_type_1 = pd.merge(left=surveys_type_1, right=survey_status[['OID', 'level1', 'level2', 'DayOfWeek']], on='OID', how='left')
surveys_type_1 = surveys_type_1[surveys_type_1['level2'] != 53]

# full week
weighted_avg = lambda x: np.average(x, weights=surveys_type_1.loc[x.index, "wh_tot_cal"])
dist_by_branch = surveys_type_1.groupby(["level1"]).agg(daily_dist=("KM_TOTAL", weighted_avg))
dist_by_branch.to_csv(folder_project / 'parameters' / 'dist_by_branch_full_week.csv')

# weekdays only
surveys_type_1_tmp = surveys_type_1[surveys_type_1['DayOfWeek'].isin(['MO', 'DI', 'MI', 'DO', 'FR'])]
weighted_avg = lambda x: np.average(x, weights=surveys_type_1_tmp.loc[x.index, "wh_tot_cal"])
dist_by_branch = surveys_type_1_tmp.groupby(["level1"]).agg(daily_dist=("KM_TOTAL", weighted_avg))
dist_by_branch.to_csv(folder_project / 'parameters' / 'dist_by_branch_weekday.csv')
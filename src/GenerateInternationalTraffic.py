import numpy as np
import pandas as pd
import openmatrix as omx
import tools

OMXfile = omx.open_file('tt_matrix_international.omx', 'r') # Travel time matrix for all NPVM zones (including those abroad), for conventional passenger cars
tt_matrix_full = np.array(OMXfile['131']).astype(np.float32) # includes also the zones 11 in Liechtenstein and 2 enclaves. It is not a problem, because these occupy the last indices (7966-7978)
mapping = OMXfile.mapping('NO')
OMXfile.close()

N =tt_matrix_full.shape[0]

## Load the explanatory variables (population and jobs per zone)
ZoneStatsCH = pd.read_csv('FTE_pro_zoneNPVM.csv', sep=";")
ZoneStatsLI = pd.read_csv('Strukturdaten_LIE_BuesCamp.csv', sep=",")
ZoneStatsAbroad = pd.read_csv('StrD_Ausland_2025_2050.csv', sep=";")

pop= np.zeros(N, dtype=np.float32)
jobs= np.zeros(N, dtype=np.float32)

pop[:7965]=ZoneStatsCH.loc[:,'Pop']
pop[7965:7978]=ZoneStatsLI.loc[:,'resident']
pop[7999:8709]=ZoneStatsAbroad.loc[:,'QZD_Einwohner_2017'] # Note: the so-called "replacements zones" (index 7979-7999, starting from 1) and the cordon zones (8710 - end) have no population and no job, so we do not generate any traffic for them.

jobs[:7965]=ZoneStatsCH.loc[:,'Industrial']+ZoneStatsCH.loc[:,'Trade']+ZoneStatsCH.loc[:,'Transport']+ZoneStatsCH.loc[:,'Services']
jobs[7965:7978]=ZoneStatsLI.loc[:,'sum_fte']
jobs[7999:8709]=ZoneStatsAbroad.loc[:,'QZD_Beschaeftigte_2017']

## Generation step
ProductionRates = pd.read_csv('parameters\\producation_rates.csv', sep=",", index_col=0)
Trips_pro_zone = pop * ProductionRates.loc['pop', 'value'] + jobs * ProductionRates.loc['jobs', 'value']

## Distribution step
# Logit parameter
GravityModel = pd.read_csv('GravityModel.csv', sep=",", index_col=0)
logit_par = GravityModel.loc['logit_par', 'value']
TripsGravity = Trips_pro_zone * np.transpose(Trips_pro_zone) * np.exp(logit_par * tt_matrix_full)
    
# Iterative Proportional Fitting
TripsGravityBalanced = tools.Furness(TripsGravity, Trips_pro_zone, Trips_pro_zone, 0.0001, 1000)

## Remove all internal trips (already taken care for by the actual LCV model)
TripsGravityBalanced[:7965,:7965]=0

## Save resulting international traffic
myfile = omx.open_file('Trips_international.omx','w')
myfile['Trips'] = TripsGravityBalanced
myfile.create_mapping('NO',list(mapping.keys()))
myfile.close()

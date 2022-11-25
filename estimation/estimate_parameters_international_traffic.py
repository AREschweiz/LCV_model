import numpy as np
import pandas as pd
import openmatrix as omx
import time
import numpy.matlib
import matplotlib.pyplot as plt
import tools
from pathlib import Path

# parent folder
path_parent = str(Path(__file__).parent.resolve())

######### Production / attraction rates
# Note:
# by construction, the number of trips departing from a zone and
# arriving in this zone are equal. Only one rate is needed.

# Load the reference OD matrix
OMXfile = omx.open_file('Trips_g0.1_4.omx', 'r')
Trips = np.array(OMXfile['Trips']).astype(np.float32) # includes also the zones 11 in Liechtenstein and 2 enclaves. It is not a problem, because these occupy the last indices (7966-7978)
Trips = np.delete(Trips, slice(7965,7978), axis=0)
Trips = np.delete(Trips, slice(7965,7978), axis=1)
OMXfile.close()

# Compute the number of trips departing from (=arriving to) each zone.
Trips_pro_zone = np.sum(Trips, axis=0).reshape(7965,1)

# Load the population and jobs pro zone
ZoneStats = pd.read_csv(path_parent+'input_data\\FTE_pro_zoneNPVM.csv', sep=";")
N = len(ZoneStats.index) # number of rows = number of zones.
Pop = ZoneStats.loc[:,'Pop'].to_numpy(dtype=np.float32).reshape(N,1)
jobs = (ZoneStats.loc[:,'Industrial']+ZoneStats.loc[:,'Trade']+ZoneStats.loc[:,'Transport']+ZoneStats.loc[:,'Services']).to_numpy(dtype=np.float32).reshape(N,1)

# Do the regression
reg_results = np.linalg.lstsq(np.concatenate((Pop, jobs), axis = 1), Trips_pro_zone, rcond = None)
r_square = 1 - reg_results[1]/sum(np.power(Trips_pro_zone,2))
coeff_pop = reg_results[0][0][0]
coeff_jobs = reg_results[0][1][0]

# write the parameters in a csv or text file.
myfile = open(path_parent+"parameters\\production_rates.csv",'w')
myfile.write("variable,value\n")
myfile.write("pop,{0:9.8f}\n".format(coeff_pop))
myfile.write("jobs,{0:9.8f}".format(coeff_jobs))
myfile.close()             
             
######### Gravity model 

OMXfile = omx.open_file('tt_matrix.omx', 'r')
tt_matrix = np.array(OMXfile['1']).astype(np.float32) # includes also the zones 11 in Liechtenstein and 2 enclaves. It is not a problem, because these occupy the last indices (7966-7978)
tt_matrix = np.delete(tt_matrix, slice(7965,7978), axis=0)
tt_matrix = np.delete(tt_matrix, slice(7965,7978), axis=1)
tt_matrix[np.diag_indices(7965,ndim=2)] = 0
OMXfile.close()

def RMSE_gravity_logit(Trips, tt_matrix, logit_par):
		# This function computes the root-mean squared error for some given parameters, which is then the objective to be minimized.
    Trips_pro_zone = np.sum(Trips, axis=0).reshape(7965,1)
    TripsGravity = Trips_pro_zone * np.transpose(Trips_pro_zone) * np.exp(logit_par * tt_matrix)
    # Apply a quick Furness
    TripsGravityBalanced = tools.Furness(TripsGravity, Trips_pro_zone, Trips_pro_zone, 0.0001, 1000)
    # Compute RMSE
    RMSE = (sum(sum(np.power(Trips - TripsGravityBalanced, 2)))/(N*N)) ** 0.5
    
    return RMSE
    
xlist = np.linspace(-0.15, -0.07,40, endpoint=False) # list of parameter values at which we will evaluate the RMSE
RMSE = np.zeros(len(xlist), dtype = np.float32) # initialization

for i in range(0, len(xlist)): # compute the RMSE for each parameter value
    time_old=time.time()
    print("i:{0}".format(i))
    RMSE[i] = RMSE_gravity_logit(Trips, tt_matrix, xlist[i])
    print(time.time()-time_old)

# plot RMSE as a function of the logit parameter, for visual check that we have found the minimum.
fig,ax=plt.subplots(1,1)
plt.plot(xlist,RMSE)
ax.set_title('Estimation of logit parameter')
ax.set_xlabel('logit parameter')
ax.set_ylabel('RMSE')
plt.show()

logit_par_best = xlist[np.argmin(RMSE)] # the best value is the one minimizing the RMSE

# save the estimated value
myfile = open(str(path) + "\\parameters\\GravityModel.csv",'w')
myfile.write("parameter,value\n")
myfile.write("logit_par,{0:5.3f}\n".format(logit_par_best))
myfile.close()   
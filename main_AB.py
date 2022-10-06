# Main script for the simulation of LCV-model (agent-based version)
# Author: Raphael Ancel, Swiss Federal Office for Spatial Development

# Import necessary libraries
import numpy as np
import pandas as pd
import time
import openmatrix as omx
import xarray
import os

# Set path to current folder.
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# ---------------------------------------------------------------------------
# Import explanatory variables and travel time matrix
# ---------------------------------------------------------------------------

# Import as DataFrame and store the most important variables separately 
ZoneStats = pd.read_csv('input_data/FTE_pro_zoneNPVM.csv', sep=";")
N = len(ZoneStats.index) # number of rows = number of zones.

area = ZoneStats.loc[:,'Area'].to_numpy(dtype=np.float32).reshape(N,1)
Pop = ZoneStats.loc[:,'Pop'].to_numpy(dtype=np.float32).reshape(N,1)
jobs = (ZoneStats.loc[:,'Industrial']+ZoneStats.loc[:,'Trade']+ZoneStats.loc[:,'Transport']+ZoneStats.loc[:,'Services']).to_numpy(dtype=np.float32).reshape(N,1)
accessibility =ZoneStats.loc[:,'Strasse_Erreichb_EWAP'].to_numpy(dtype=np.float32).reshape(N,1)

# Determine the type of land-use (see technical report for definition)
LandUse = np.tile("L", (N,1))
for zone in range(0,N):
    if Pop[zone,0]/area[zone,0] <=100 and jobs[zone,0]/area[zone,0] <=100:
        LandUse[zone,0] = "L" # Low-density area
    elif Pop[zone,0]/area[zone,0] > 100 and jobs[zone,0] <= 2 * Pop[zone,0]:
        LandUse[zone,0] = "R" # Residential area
    elif jobs[zone,0]/area[zone,0] <= 3000:
        LandUse[zone,0] = "I" # Industrial area
    else:
        LandUse[zone,0] = "E" # Employment node

# Import travel time matrix
OMXfile = omx.open_file('input_data/tt_matrix.omx', 'r')
tt_matrix = np.array(OMXfile['1']).astype(np.float32)
# remove the  11 zones in Liechtenstein and 2 enclaves, which occupy the last indices.
tt_matrix = np.delete(tt_matrix, slice(7965,7978), axis=0)
tt_matrix = np.delete(tt_matrix, slice(7965,7978), axis=1)
tt_matrix[np.diag_indices(7965,ndim=2)] = 0 # forces the travel times for internal trips to be 0.
mapping = OMXfile.mapping('NO')
OMXfile.close()


# ---------------------------------------------------------------------------
# Model application
# ---------------------------------------------------------------------------

#### Vehicle Generation
# Input = number of jobs (FTE) per branch + population, both at zone level (National Passenger Transport Model 2017)
# Parameters : vehicle rates per branch and for the population, for each segment
# Segmentation : CurbWeight up to 2 tons / above 2 tons
# Output = Nx2x(9+1) array of number of vehicles pro zone, CurbWeight and branch (float)

# read parameters
GenPar = pd.read_csv('parameters/VehicleGeneration.csv', sep=",",index_col=0)
branches = GenPar.columns.tolist() # list of branches used (includes population as a branch)
veh_types = GenPar.index


# initialize the output
Veh = xarray.DataArray(np.empty((N,2,10), dtype=object), coords = [range(0,N), veh_types, branches], dims = ["zones", "veh_types", "branches"])

# Apply the model, i.e. multiply the vehicle rates by the corresponding explanatory variable for each vehicle type and each branch (and each zone).
for veh_type in veh_types:
    for branch in branches:
        Veh.loc[:, veh_type, branch] = ZoneStats.loc[:, branch] * GenPar.loc[veh_type, branch]
print('Vehicle generation: done')

####### Vehicle Purpose
# Mapping used for convenience purpose, because vehicles generated for the branch "Pop" (i.e. privately-owned vehicles) use the parameters estimated for vehicles whose branch is ``unknown''.
map_branch_estimation = {}
for branch in branches:
    map_branch_estimation[branch] = branch
map_branch_estimation['Pop'] = 'Unknown'
# inverse of map_branch_estimation
inv_map = {v: k for k, v in map_branch_estimation.items()}

# read parameters
PurposePar = pd.read_csv('parameters/VehiclePurpose.csv', sep=";")
purposes = PurposePar.columns[2:] # list of purposes

# Apply proportion of active vehicles to the number of vehicles.
ActiveShares = pd.read_csv('parameters/ShareActive.csv', sep=";", index_col=0)
for veh_type in veh_types:
    for branch in branches:
        Veh.loc[:, veh_type, branch] = Veh.loc[:, veh_type, branch] * ActiveShares.loc[map_branch_estimation[branch],'Share Active']

# initialize the output
VehByPurpose=xarray.DataArray(np.empty((N,2,10,3), dtype=object), coords = [range(0,N), veh_types, branches, purposes], dims = ["zones", "veh_types", "branches", "purposes"])
# Distribute the active vehicles between the 3 purposes
for i in PurposePar.index:
    for purpose in purposes:
        VehByPurpose.loc[:, PurposePar.loc[i, 'CurbWeight'], inv_map[PurposePar.loc[i, 'Halter']], purpose] = PurposePar.loc[i, purpose] * Veh.loc[:, PurposePar.loc[i, 'CurbWeight'], inv_map[PurposePar.loc[i, 'Halter']]]
print('Vehicle Purpose: done')

####### Number of tours
ToursPar = pd.read_csv('parameters/NumberOfTours.csv', sep=";")
# initialize the output
Tours_df = xarray.DataArray(np.empty((N,2,10,3), dtype=object), coords = [range(0,N), veh_types, branches, purposes], dims = ["zones", "veh_types", "branches", "purposes"])
for i in ToursPar.index:
    if ToursPar.loc[i, 'NOGA_1'] in map_branch_estimation.values(): #to discard parameters corresponding to branches for which no vehicle is generated.
        Tours_df.loc[:, ToursPar.loc[i, 'CurbWeight'], inv_map[ToursPar.loc[i, 'NOGA_1']], ToursPar.loc[i, 'Purpose']] = VehByPurpose.loc[:, ToursPar.loc[i, 'CurbWeight'], inv_map[ToursPar.loc[i, 'NOGA_1']], ToursPar.loc[i, 'Purpose']] * ToursPar.loc[i, 'AvgNbTours']
print('Number of tours: done')

####### Correction
# Load correction factors
CorrectionFactors = pd.read_csv('parameters/CorrectionFactors.csv', sep=';', index_col=0) # we could also use the csv for NextStopLocation as the segment definition is the same
# Apply them to each segment
for purpose in purposes:
    for veh_type in veh_types:
        for branch in branches:
            Tours_df.loc[:,veh_type,branch,purpose] = Tours_df.loc[:,veh_type,branch,purpose] * CorrectionFactors.loc[map_branch_estimation[branch],veh_type]
print('Correction: done')

####### Conversion to segments for Tour Building
EndTourPar = pd.read_csv('parameters/EndTour.csv', sep=';') # we could also use the csv for NextStopLocation as the segment definition is the same
Tours_np = np.zeros((N,len(EndTourPar.index)), dtype=np.float32)
for purpose in purposes:
    for veh_type in veh_types:
        for branch in branches:
            j = 0
            while True: #for each combination of purpose x veh_type x branch, find the segment (i.e. row of EndTourPar) that matches this segment.
                if (EndTourPar.loc[j, 'Purpose'] == purpose) & (EndTourPar.loc[j, 'CurbWeight'] in [veh_type, 'X']) & (EndTourPar.loc[j, map_branch_estimation[branch]] == 1):
                    Tours_np[:,j] = Tours_np[:,j] + Tours_df.loc[:, veh_type, branch, purpose].values
                    break #segment was found, exit loop
                j = j+1 #otherwise, try next segment
print('Segment aggregation: done')
# Check: The number of tours in the segments 6 (Goods, light, H) and 12 (Goods, heavy, N) should be 0 (because the generation rates are 0 for light x H and heavy x N)

####### Tour building
Tours_by_seg=sum(Tours_np)
Trips = np.zeros((N, N)).astype(np.float32)
NextLocPar = pd.read_csv("parameters/NextStopLocation.csv", delimiter = ';') 

granularity=0.1 # defines how "big" the modelled agents are. 
# A granularity of 0.1 means that if in reality 100 tours are generated from a zone on a representative day,
# then 100/0.1= 1000 tours are generated in simulations, each of them counting for 0.1 vehicle equivalent.
# The smaller the granularity, the longer the computational time and the smaller the variability between simulations.

time_start = time.time() # This is just to measure computational time.
# To have an idea of the total computational time (approximately linear to the number of tours simulated)
Total_tours_to_simulate=np.sum(Tours_np)/granularity
Total_simulated_tours=0
Next_displayed_threshold=1/100*Total_tours_to_simulate
# Tour building:
for ind_seg in range(0, len(EndTourPar.index)):# for each segment
    print("segment: " + str(ind_seg))
    if Tours_by_seg[ind_seg]>0:
        # pre-compute destination component for NextStopLocation
        destination_component = NextLocPar.at[ind_seg, 'b_LowDen'] * (LandUse == "L") + \
        NextLocPar.at[ind_seg, 'b_Res'] * (LandUse == "R") + \
        NextLocPar.at[ind_seg, 'b_Inter'] * (LandUse == "I") + \
        NextLocPar.at[ind_seg,'b_EmpNode'] * (LandUse == "E") + \
        NextLocPar.at[ind_seg, 'b_size'] * \
        np.log(1 + Pop + NextLocPar.at[ind_seg, 'b_jobs'] * (jobs))
        destination_component = np.reshape(destination_component,(1,N)).astype(np.float32)
        
        # preliminary computations for EndTour
        exp_U_return = np.ones((N,1)).astype(np.float32) # Utility to return is taken as reference (1)
        # pre-compute the part of the utility to continue (i.e. server one more stop) which is only zone dependent
        U_continue_0 = (EndTourPar.at[ind_seg, 'ASC'] + \
            EndTourPar.at[ind_seg, 'BSC'] + \
                EndTourPar.at[ind_seg, 'LargeCurbWeight'] + \
        EndTourPar.at[ind_seg, 'b_access'] * accessibility/40000).astype(np.float32)
        
        for o in range(0,N): # iterates over origins
            if Total_simulated_tours > Next_displayed_threshold: # prints progress (only checks progress once per origin zone, to limit overhead)
                time_now=time.time()
                print("progress : {prog:.0%}, elapsed time = {x:.2f} s".format(prog=Total_simulated_tours/Total_tours_to_simulate, x= time_now-time_start))
                Next_displayed_threshold = Next_displayed_threshold + 1/100*Total_tours_to_simulate

            ### stochastic part (iteration over agents) starts here:
            for i in range(0,int(np.round(Tours_np[o, ind_seg]/granularity,decimals=0))): # the number of generated agents is rounded and depends on granularity
                curr_zone = o #current zone
                legRank = 0 # leg rank in the tour
                EndTour = False
                while not EndTour:
                    ######## Next Stop Location
                    # first compute the utility of each zone
                    if legRank== 0:
                        U = destination_component + NextLocPar.at[ind_seg, 'b_tt_0_first'] * tt_matrix[curr_zone,:]
                    else:
                        U = destination_component + NextLocPar.at[ind_seg, 'b_tt_0'] * tt_matrix[curr_zone,:] + \
                            np.transpose(NextLocPar.at[ind_seg, 'b_tt_target_to_base'] * tt_matrix[:,o])
                    U = U + NextLocPar.at[ind_seg, 'b_tt_20'] * np.maximum(tt_matrix[curr_zone,:] - 20, np.zeros((1,N),dtype = np.float32)) + \
                        NextLocPar.at[ind_seg, 'b_tt_40'] * np.maximum(tt_matrix[curr_zone,:] - 40, np.zeros((1,N),dtype = np.float32))
                    U[0,curr_zone] = U[0,curr_zone] + NextLocPar.at[ind_seg, 'b_intra']
                    expU = np.exp(U)
                    # converts utility into a probability to be chosen with a simple Multinomial logit.
                    propLocation = expU / np.sum(expU)
                    # sample the decision (=stochastic part)
                    next_zone = np.random.choice(range(0,N), p = propLocation.reshape(N))
                    # add the chosen trip leg to the overall Trip matrix
                    Trips[curr_zone, next_zone] = Trips[curr_zone, next_zone] + granularity
                    # updates the current zone and leg rank
                    curr_zone = next_zone
                    legRank = legRank + 1                
                    
                    ######## EndTour
                    # Add the missing terms to the utility to continue
                    U_continue = U_continue_0[curr_zone, 0] + EndTourPar.at[ind_seg,'b_tt_return'] * tt_matrix[curr_zone, o] + EndTourPar.at[ind_seg, 'b_lnStops'] * np.log(legRank + 1)
                    if legRank == 1:
                        U_continue = U_continue + EndTourPar.at[ind_seg,'cons_2stops']
                    # samples the decision to continue (=stochastic part)
                    seed = np.random.rand()
                    if seed < 1/(1+np.exp(U_continue)): # decides to returns to home establishment
                        EndTour = True 
                        Trips[curr_zone, o] = Trips[curr_zone, o] + granularity # adds the return trip to the overall Trip matrix.
                        Total_simulated_tours = Total_simulated_tours + 1
       
time_now=time.time()
print("progress : 100 % ; Time needed for tour construction:  {x:.2f} s".format(x=time_now-time_start))
# augments the size of the trip matrix to include zones in Liechtenstein as well as Büsingen and Campione d'Italia (OD pairs filled with zeros)
Trips7978=np.zeros((7978,7978), dtype= np.float32)
Trips7978[:7965,:7965]=Trips
# writes the matrix in the OMX format, compatible with VISUM
myfile = omx.open_file('Trips_g' + str(granularity) + '.omx','w')
myfile['Trips'] = Trips7978
myfile.create_mapping('NO',list(mapping.keys()))
myfile.close()

  
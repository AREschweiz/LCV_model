# This module loads the data from the national vehicle register and does linear regressions with the population and the number of jobs per branch
#import statsmodels.formula.api as sm
import pandas as pd
import numpy as np

# load vehicle register
veh_register = pd.read_csv('BEST_R-20220601.txt', sep="\t")
#a=veh_register.columns
# remove unnecessary vehicle categories
veh_register = veh_register[veh_register["Fahrzeugart_Code"].isin([30,36,38])]
# remove too heavy vehicles
veh_register = veh_register[veh_register["Gesamtgewicht"]<=3500]
# keep only Swiss ones
veh_register = veh_register[veh_register["Staat_Code"]=='CH']

# load the number of jobs and inhabitant per PLZ and branch
PLZ_variables = pd.read_csv('pop_and_job_pro_PLZ_and_NOGA.csv', sep=";", index_col=0)

# add a column to veh_register indicating whether the vehicle is light (curbweight <=2 tons)
veh_register["is_light"]=veh_register["Leergewicht"]<=2000
# add a column to veh_register indicating whether the vehicle is privately-owned
veh_register["is_private"]=veh_register["Halterart_Code"]<3 # 1: male, 2: female, 3: business, 4: unknown (-> assumed to be business)

veh_stats = pd.pivot_table(veh_register,values="Leergewicht",index="is_light",columns=["is_private"],aggfunc=len,fill_value=0)

# aggregate vehicle per PLZ and according to the segmentation criteria (Curbweight and owner type)
veh_agg_PLZ = pd.pivot_table(veh_register,values="Leergewicht",index="PLZ",columns=["is_light","is_private"],aggfunc=len,fill_value=0)

# discard the PLZ which do not belong to Switzerland (Liechtenstein mostly, but also many PLZ do not corresponding to zones - 3003 Bern for instance) - THIS IS NOT GOOD.
veh_agg_PLZ=veh_agg_PLZ[~(veh_agg_PLZ.index.isin(PLZ_variables.index))]

# merge the two DataFrames
PLZ_variables=pd.concat([PLZ_variables,veh_agg_PLZ],axis=1)
PLZ_variables=PLZ_variables.rename(columns={(False,False):'heavy_Business',  (True,False):'light_Business',(False,True):'heavy_Private',(True,True):'light_Private'})

# Do the regressions

# Business-owned Light LCV
result_LB = sm.ols(formula="light_Business ~ A + B + C + D + E + F + G + N -1", data=PLZ_variables).fit() # the "-1" means "without intercept"
#print(result_LB.params)
print(result_LB.summary())

# Business-owned Heavy LCV
result_HB = sm.ols(formula="heavy_Business ~ A + B + C + D + E + F + G + H -1", data=PLZ_variables).fit() # the "-1" means "without intercept"
#print(resul_HBt.params)
print(result_HB.summary())

# Privately-owned Light LCV
result_LP = sm.ols(formula="light_Private ~ Pop -1", data=PLZ_variables).fit() # the "-1" means "without intercept"
#print(result_LP.params)
print(result_LP.summary())

# Privately-owned Light LCV
result_HP = sm.ols(formula="heavy_Private ~ Pop -1", data=PLZ_variables).fit() # the "-1" means "without intercept"
#print(result_HP.params)
print(result_HP.summary())

# save resulting coefficients as vehicle generation rates
#explanatory_variables=np.unique(np.concatenate((result_LB.params.index.values, result_HB.params.index.values , result_LP.params.index.values, result_HP.params.index.values)))
#rates = pd.DataFrame(data=0,index=['light','heavy'],columns=explanatory_variables,dtype=np.float32)
#rates.loc['light',:]=rates.loc['light',:]+result_LB.params.to_frame().T+result_LP.params.to_frame().T
results_light=pd.concat([result_LB.params.to_frame().T,result_LP.params.to_frame().T],axis=1)
results_light=results_light.rename(index={0:'Light'})
results_heavy=pd.concat([result_HB.params.to_frame().T,result_HP.params.to_frame().T],axis=1)
results_heavy=results_heavy.rename(index={0:'Heavy'})
results=pd.concat([results_light,results_heavy],axis=0).fillna(0)
results.to_csv('VehicleGeneration.csv',sep=',')






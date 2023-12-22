###################
# This script generates a csv file containing various socioeconomic variable for each zone of the transport model
# To run it, one needs to have access to the STATPOP (aggregated by transport zone) and STATENT data (point-data).
###################

import pandas as pd
import geopandas
import numpy as np
from typing import Any, Dict, List, Tuple
from pathlib import Path
from src.support import get_omx_matrix

year_data = 2021
folder_project = Path.cwd().parent

path_STATPOP = f'D:/GIS/STATPOP/STATPOP{year_data}_ZonenNPVM.shp' # shapefile containing the population by zone
path_STATENT = f'D:/GIS/STATENT/STATENT_{year_data}.gpkg' # gpkg containing the STATENT (point-data)

# Load Statent
gdf = geopandas.read_file(path_STATENT).to_crs(2056)

# Load NOGA mapping
NOGA_mapping = pd.read_csv(folder_project / 'data' / 'NOGA_map_level_1_5.csv', sep=';')
NOGA_mapping.rename(mapper={'Type':'NOGA_CD_2008_6'}, axis=1, inplace=True)
NOGA_mapping.drop('Name_en', axis=1, inplace=True)
# Load Population with NPVM geometry
#zones = geopandas.read_file('D:/PycharmProjects/LCV_model_uebergabe/inputs/Verkehrszonen_Schweiz_NPVM_2017.shp')
zones = geopandas.read_file(path_STATPOP).to_crs(2056)
zones['Area'] = zones['geometry'].area / (10**6)

# Do merge to augment the Statent Dataframe with the NOGA sections.
gdf = gdf.merge(NOGA_mapping, on="NOGA_CD_2008_6")

# Do spatial join to augment the Statent Dataframe with the zone NPVM
gdf = geopandas.sjoin(gdf, zones)

# groupby zone, NOGA., sum over FTE.
gdf = gdf[['ID', 'geometry', 'Section', 'EMPFTE']] # drop columns we don't need

stats_NPVM = gdf.dissolve(by=['ID', 'Section'], aggfunc='sum')
stats_NPVM = pd.DataFrame(stats_NPVM.drop(columns='geometry'))
stats_NPVM = stats_NPVM.unstack(level='Section', fill_value=0)
stats_NPVM = stats_NPVM.droplevel(level=0, axis=1)
stats_NPVM['Jobs'] = stats_NPVM.sum(axis=1)

# Ensure that also the zones without any job are part of the file
zones = zones.set_index('ID')
stats_NPVM = pd.merge(left=stats_NPVM, right=zones[['Population', 'Area']], left_index=True, right_index=True, how='outer')
stats_NPVM = stats_NPVM.fillna(0)

# remove zones outside CH
stats_NPVM = stats_NPVM.loc['101001':'681001005']

# Add road accessibility
beta: float = 0.05
tt_matrix, zone_mapping, zone_ids = get_omx_matrix(str(folder_project / 'data' / 'DWV_2017_Strasse_Reisezeit_CH_7978zones.omx'), n_zones=7965, n_external_zones= 13)
stats_NPVM['RoadAccessibility'] = np.matmul(np.exp(-beta*tt_matrix), stats_NPVM['Population'].values) + 0.5 * np.matmul(np.exp(-beta*tt_matrix), stats_NPVM['Jobs'].values)
stats_NPVM = stats_NPVM.rename(columns={'Population':'Pop'})

# Export
stats_NPVM.to_csv(folder_project / 'data' / f'zone_stats_{year_data}_NPVM.csv', sep=";")


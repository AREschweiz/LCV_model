#############
# This script makes a map of the demand and of the delivery tours of a chosen courier
#############

import matplotlib.pyplot as plt
import pandas as pd
import shapefile as shp
from descartes import PolygonPatch

from pathlib import Path
folder_project = Path.cwd().parent

path_zones = str(folder_project / 'data' / 'Verkehrszonen_Schweiz_NPVM_2017.shp')
path_depots = str(folder_project / 'parameters' / 'ParcelDepots.csv')

path_demand = folder_project / 'outputs' / 'run2013' / 'parcel_demand.csv'
path_schedules = folder_project / 'outputs' / 'run2013' / 'parcel_schedules.csv'

n_zones_internal = 7965

select_courier = 'La Poste'

#%%

print('Reading data...')

with shp.Reader(path_zones, encodingErrors="ignore") as sf:
    zones_geometry = sf.__geo_interface__

depots = pd.read_csv(path_depots, sep=';')

demand = pd.read_csv(path_demand, sep=';')
schedules = pd.read_csv(path_schedules, sep=';')


#%%

print('Creating demand plot...')

fig = plt.figure(figsize=(10, 10))
ax = fig.gca()

# Plot the zones
for i, feature in enumerate(zones_geometry['features']):
    if i < n_zones_internal:
        ax.add_patch(PolygonPatch(feature['geometry'], fc='#cccccc', ec='#ffffff', linewidth=0.1, zorder=0))
ax.axis('scaled')

# Plot the demand
courier_demand = demand[demand['courier'] == select_courier]
for row in courier_demand.to_dict('records'):
    ax.plot(
        [row['orig_x_coord'], row['dest_x_coord']], [row['orig_y_coord'], row['dest_y_coord']],
        lw=0.5, zorder=1, color='#000080')

# Plot the depots
courier_depots = depots[depots['Courier'] == select_courier]
ax.scatter(
    courier_depots['X'], courier_depots['Y'],
    s=10, zorder=2, color='b')

ax.set_xticks([])
ax.set_yticks([])

print('Exporting plot...')

plt.savefig(folder_project / 'outputs' / 'figures' / f'parcel_demand_{select_courier.lower()}.png', bbox_inches='tight', dpi=300)


#%%

print('Creating schedules plot...')

fig = plt.figure(figsize=(10, 10))
ax = fig.gca()

# Plot the zones
for i, feature in enumerate(zones_geometry['features']):
    if i < n_zones_internal:
        ax.add_patch(PolygonPatch(feature['geometry'], fc='#cccccc', ec='#ffffff', linewidth=0.1, zorder=0))
ax.axis('scaled')

# Plot the schedules
courier_schedules = schedules[schedules['courier'] == select_courier]
for row in courier_schedules.to_dict('records'):
    ax.plot(
        [row['orig_x_coord'], row['dest_x_coord']], [row['orig_y_coord'], row['dest_y_coord']],
        lw=0.3, zorder=1, color='#000080')

# Plot the depots
courier_depots = depots[depots['Courier'] == select_courier]
ax.scatter(
    courier_depots['X'], courier_depots['Y'],
    s=10, zorder=2, color='b')

ax.set_xticks([])
ax.set_yticks([])

print('Exporting plot...')

plt.savefig(folder_project / 'outputs' / 'figures' / f'parcel_schedules_{select_courier.lower()}.png', bbox_inches='tight', dpi=300)

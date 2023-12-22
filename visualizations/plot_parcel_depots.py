################
# This script makes a map with all parcel depots locations (colored by courier)
################
import matplotlib.pyplot as plt
import pandas as pd
import shapefile as shp
from descartes import PolygonPatch
from pathlib import Path
folder_project = Path.cwd().parent

path_zones = str(folder_project / 'data' / 'Verkehrszonen_Schweiz_NPVM_2017.shp')
path_depots = str(folder_project / 'parameters' / 'ParcelDepots.csv')

n_zones_internal = 7965

courier_to_color = {
    'La Poste': 'r',
    'FedEx': '#cc0099',
    'DPD': 'k',
    'DHL': '#ff9900'}

print('Reading data...')

with shp.Reader(path_zones, encodingErrors="ignore") as sf:
    zones_geometry = sf.__geo_interface__

depots = pd.read_csv(path_depots, sep=';')

print('Creating plot...')

fig  = plt.figure(figsize=(10, 10))
ax = fig.gca()

# Plot the zones
for i, feature in enumerate(zones_geometry['features']):
    if i < n_zones_internal:
        ax.add_patch(PolygonPatch(feature['geometry'], fc='#cccccc', ec='#ffffff', linewidth=0.1, zorder=0))
ax.axis('scaled')

# Plot the depots
for courier in depots['Courier'].unique():
    courier_depots = depots[depots['Courier'] == courier]
    ax.scatter(courier_depots['X'], courier_depots['Y'], s=4, zorder=1, color=courier_to_color[courier], label=courier)

ax.set_xticks([])
ax.set_yticks([])

plt.legend()

print('Exporting plot...')

plt.savefig(folder_project / 'outputs' / 'figures' / 'depots.png', bbox_inches='tight', dpi=300)

## Import necessary libraries
import numpy as np
import pandas as pd
import openmatrix as omx
import pickle
from dash import Dash, dcc, html, Input, Output, dash_table, callback
import dash_mantine_components as dmc
import plotly.express as px
import plotly.graph_objects as go
import geojson
import geopandas
import matplotlib.pyplot as plt
import itertools

## Load the OD matrix
OMXfile = omx.open_file('Trips_g0.1.omx', 'r')
Trips = np.array(OMXfile['Trips']).astype(np.float32) # trip OD matrix
mapping = OMXfile.mapping('NO')
Zone_IDs = np.array(list(mapping.keys())) # Zone IDs used in the OD matrix
OMXfile.close()

# set some start values for debugging (only used if plotting outside Dash)
direction='in' # incoming or outgoing flows
zone= Zone_IDs[0] # zone of interest; for which we want to plot the flows.
ind_zone=np.where(Zone_IDs==zone)

# first solution with geopandas (not compatible with dash)
# gpd=geopandas.read_file("input_data/zonesNPVM_4326.geojson")
# gj_from_gpd=gpd.to_json() # warning, this does not produce the same result as reading the geojson file directly (as done below)
# gpd.set_index("ID",inplace=True)
# gpd.fillna(0)
# gpd=pd.concat([gpd,pd.DataFrame(data=Trips[:, ind_zone].reshape(7978, 1),index=Zone_IDs,columns=['flow'])],axis=1)
# gpd.plot(column='flow',aspect=1)
#plt.show()

# Solution with plotly express
with open("input_data/zonesNPVM_4326.geojson") as f: # load zones geometry. Important: zones must be in CRS WGS84 and the file must be open using geojson.load. converting a geopandas file does not work as this will not include the info about CRS.
    gj = geojson.load(f)

# first without dash, to facilitatate debugging
# if direction == 'in': # incoming flows
#     flows = Trips[:, ind_zone].reshape(7978, 1)
# else: # outgoing flows
#     flows = Trips[ind_zone, :].reshape(7978, 1)
# # store data in Dataframe together with zone IDs
# df = pd.DataFrame(data=np.concatenate([Zone_IDs.reshape(-1, 1), flows], axis=1),
#                   columns=['ZoneID', 'flow'])
# # to accelerate image rendering, we only plot zones with positive flows (otherwise it is very slow)
# df_without_0 = df[df.flow !=0] # remove zones with zero flow from Dataframe
# IDs_with_0 = Zone_IDs[(flows == 0).reshape(-1)]
# gj_without_0 = gj.copy() # copy geojson file before removing zones
# features = []
# for k, v in itertools.groupby(
#         [x for x in gj['features'] if not (x['properties']['ID'] in IDs_with_0)]):  # remove zones with zero flow
#     features.append(k)
# gj_without_0['features'] = features
#
# fig = px.choropleth_mapbox( # plot heatmap
#     df_without_0, geojson=gj_without_0, color="flow",
#     locations='ZoneID', featureidkey="properties.ID",
#     opacity=0.5,
#     range_color=[0, max(flows)[0]],
#     zoom=8, center={"lat": 47, "lon": 8})
# fig.update_layout(mapbox_style="open-street-map")
# fig.show()


# version with Dash.
app = Dash(__name__)

app.layout = html.Div([
    html.H4('LCV Flows analysis from/to a given zone'),
    html.P("Select a zone and direction:"),
    dcc.Dropdown(
        id='zone',
        options=Zone_IDs,
        value=Zone_IDs[0]), # initial value
    dcc.Dropdown(
        id='direction', #incoming or outgoing
        options=['in','out'],
        value='in'), # initial value
    dcc.Graph(id="graph",style={'width':'90vw', 'height': '90vh'}),
])

@app.callback(
    Output("graph", "figure"),
    Input("zone", "value"),
    Input("direction", "value"),)

def make_map(zone,direction):
    ind_zone = np.where(Zone_IDs == zone)
    if direction == 'in': #incoming flows
        flows = Trips[:, ind_zone].reshape(7978, 1)
    else: #outgoing flows
        flows = Trips[ind_zone, :].reshape(7978, 1)
    # save data in dataframe with zone IDs
    df = pd.DataFrame(data=np.concatenate([Zone_IDs.reshape(-1, 1), flows], axis=1),
                      columns=['ZoneID', 'flow'])
    # remove zones with no flow
    df_without_0 = df[df.flow != 0] # first remove them from Dataframe
    IDs_with_0 = Zone_IDs[(flows == 0).reshape(-1)]
    gj_without_0 = gj.copy() # then from geojson
    features = []
    for k, v in itertools.groupby(
            [x for x in gj['features'] if not (x['properties']['ID'] in IDs_with_0)]):  # filter out zones with 0 flow
        features.append(k)
    gj_without_0['features'] = features
    # heatmap
    fig = px.choropleth_mapbox(
        df_without_0, geojson=gj_without_0, color="flow",
        locations='ZoneID', featureidkey="properties.ID",
        opacity=0.5,
        range_color=[0, max(flows)[0]],
        zoom=8, center={"lat": 47, "lon": 8})
    fig.update_layout(mapbox_style="open-street-map")
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)

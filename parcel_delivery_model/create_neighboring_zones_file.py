import numpy as np
import pandas as pd
import shapefile as shp

from shapely.geometry import Polygon, MultiPolygon
from typing import Any, Dict, List, Tuple


def read_shape(
    shape_path: str,
    return_geometry: bool,
    encoding: str = 'latin1'
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Reads the contents of a shapefile.
    Args:
        shape_path (str): The path of the shapefile.
        return_geometry (bool): If False, only read the contents of the DBF file as a DataFrame. If True,
            also read the geometries as a list of dictionaries.
        encoding (str, optional): Defaults to 'latin1'.
    Returns:
        Tuple[pd.DataFrame, List[Dict[str, Any]]]: A tuple, containing the following:
            - pd.DataFrame: The records of the DBF file.
            - List[Dict[str, Any]]: The geometries of the records. If return_geometry is False, an empty list if returned.
    """
    sf = shp.Reader(shape_path, encoding=encoding)
    records = sf.records()

    if return_geometry:
        geometry = sf.__geo_interface__['features']
        geometry = [geometry[i]['geometry'] for i in range(len(geometry))]
    else:
        geometry = []

    fields = sf.fields
    sf.close()

    # Get information on the fields in the DBF
    columns = [x[0] for x in fields[1:]]
    col_types = [x[1:] for x in fields[1:]]
    n_records = len(records)

    # Check for headers that appear twice
    for col in range(len(columns)):
        name = columns[col]
        where_name = [i for i in range(len(columns)) if columns[i] == name]
        if len(where_name) > 1:
            for i in range(1, len(where_name)):
                columns[where_name[i]] = (
                    str(columns[where_name[i]]) + '_' + str(i))

    # Put all the data records into a NumPy array (much faster than Pandas DataFrame)
    shape = np.zeros((n_records, len(columns)), dtype=object)
    for i in range(n_records):
        shape[i, :] = records[i][0:]

    # Then put this into a Pandas DataFrame with the right headers and data types
    shape = pd.DataFrame(shape, columns=columns)
    for col in range(len(columns)):
        if col_types[col][0] == 'C':
            shape[columns[col]] = shape[columns[col]].astype(str)
        else:
            shape.loc[pd.isna(shape[columns[col]]), columns[col]] = -99999
            if col_types[col][-1] > 0:
                shape[columns[col]] = shape[columns[col]].astype(float)
            else:
                shape[columns[col]] = shape[columns[col]].astype(int)

    if return_geometry:
        return (shape, geometry)
    else:
        return shape


if __name__ == '__main__':

    path_zone_shape = 'P:/Projects_Active/22064 ARE Audit and update of Swiss national LCV model/Work/Data/Zones/zonesNPVM.shp'

    n_external_zones = 13

    print('Reading zoning shapefile...')

    zone_dbf, zone_geometry = read_shape(path_zone_shape, True)
    zone_dbf = zone_dbf[:-n_external_zones]
    zone_geometry = zone_geometry[:-n_external_zones]
    n_zones = len(zone_dbf)

    zone_shapely = []
    for row in zone_geometry:
        if row['type'] == 'MultiPolygon':
            zone_shapely.append(
                MultiPolygon(
                    [Polygon(row['coordinates'][i][0]) for i in range(len(row['coordinates']))]))
        elif row['type'] == 'Polygon':
            zone_shapely.append(Polygon(row['coordinates'][0]))
        else:
            raise Exception(f"Expected either 'Polygon' or 'MultiPolygon' but got geometry of type '{row['type']}'.")

    zone_mapping_inv = dict((i, zone_dbf.at[i, 'ID']) for i in zone_dbf.index)

    print('Searching for neighboring zones...')

    neighboring_zones = []
    for i in range(n_zones):
        for j in range(n_zones):
            if i != j:
                if zone_shapely[i].touches(zone_shapely[j]):
                    neighboring_zones.append([zone_mapping_inv[i], zone_mapping_inv[j]])
        if i % 500 == 0:
            print('\tZone', i)
    print('\tZone', i)

    neighboring_zones = pd.DataFrame(np.array(neighboring_zones), columns=['ZONE_1', 'ZONE_2'])

    print('Exporting CSV...')

    neighboring_zones.to_csv('neighborsNPVM.csv', sep=';', index=False)

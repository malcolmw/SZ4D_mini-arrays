import geopandas
import pandas as pd
import obspy.clients.fdsn
import shapely

EVENTS_FILE = '/home/malcolmw/proj/sz4d/data/events/events_1980-2023.hdf5'
CHILE_SHAPE_FILE = '/home/malcolmw/proj/sz4d/data/geodata/WB_countries_Admin0_10m/WB_countries_Admin0_10m.shp'
CSN_STATION_FILE = '/home/malcolmw/proj/sz4d/data/stations.hdf5'

TAB_BLUE   = '#1f77b4'
TAB_ORANGE = '#ff7f0e'
TAB_GREEN  = '#2ca02c'
TAB_RED    = '#d62728'
TAB_PURPLE = '#9467bd'
TAB_BROWN  = '#8c564b'
TAB_PINK   = '#e377c2'
TAB_GRAY   = '#7f7f7f'
TAB_OLIVE  = '#bcbd22'
TAB_CYAN   = '#17becf'

CHILE_GEOM = geopandas.read_file(
    CHILE_SHAPE_FILE
).set_index(
    'NAME_EN'
).loc['Chile', 'geometry']

TRENCH_TRACE = shapely.geometry.LineString([
    [-71.750, -27.00],
    [-72.000, -28.00],
    [-72.200, -29.00],
    [-72.500, -30.00],
    [-72.600, -31.00],
    [-72.750, -32.00],
    [-72.800, -33.00],
    [-73.000, -34.00]
])

def load_events(as_geodataframe=True):
    events = pd.read_hdf(EVENTS_FILE)
    events['serial_id'] = events.index
    if as_geodataframe is True:
        events = geopandas.GeoDataFrame(
            events,
            geometry=geopandas.points_from_xy(events['longitude'], events['latitude'])
        )
    return events

def load_csn_stations(as_geodataframe=True):
    stations = pd.read_hdf(CSN_STATION_FILE)
    if as_geodataframe is True:
        stations = geopandas.GeoDataFrame(
            stations,
            geometry=geopandas.points_from_xy(stations['longitude'], stations['latitude'])
        )
    return stations
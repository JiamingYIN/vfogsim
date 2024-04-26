# %%
import pickle
import pandas as pd
import utm
data = pd.read_pickle('../data/traffic_flow/volume.pkl')
master_data = pd.read_pickle('../data/traffic_flow/masterdata.pkl')
print(data.shape)
print(master_data.head(5))
print(master_data.shape)
print(master_data.columns)

# %%
master_data.dtypes
# %%
netOffset = [-385174.69, -6671039.23] 
convBoundary = [0.00, 0.00, 1135.02, 1149.85] 
origBoundary = [24.914235, 60.156132, 24.966609, 60.189319]
cropped_master_data = master_data[(master_data['start_lat'] >= origBoundary[1]) &
                                  (master_data['start_lon'] >= origBoundary[0]) &
                                  (master_data['end_lon'] <= origBoundary[2]) &
                                  (master_data['end_lat'] <= origBoundary[3])]
cropped_master_data.head(2)
# %%
# Crop road network data
data['time'] = data.index
data['time'] = pd.to_datetime(data['time'])
selected_data = data[(data['time'] >= pd.to_datetime('2020-01-15 00:00:00+00:00'))&
                     (data['time'] < pd.to_datetime('2020-01-16 00:00:00+00:00'))]

selected_data = selected_data[list(cropped_master_data.index)]
selected_data.to_pickle('../data/traffic_flow/cropped_data.pkl')
# %%
volume_total = selected_data.sum(axis=1)
import matplotlib.pyplot as plt
plt.plot(list(volume_total))
plt.grid()

# %%
import xml.etree.ElementTree as ET

tree = ET.parse('../data/sumo_20vehicles/helsinki.net.xml')
import pandas as pd
root = tree.getroot()
edges = []
for edge in root.iter('edge'):
    edge_attrs = edge.attrib
    if 'type' in edge_attrs.keys():
        lane = edge.getchildren()[0].attrib
        allow_taxi = 0
        if 'allow' in lane.keys():
            allow = set(lane['allow'].split(' '))
            if 'taxi' in allow:
                allow_taxi = 1
        if 'disallow' in lane.keys():
            disallow = set(lane['disallow'].split(' '))
            if 'taxi' not in disallow:
                allow_taxi = 1

        edge_dict = {'id': edge_attrs['id'],
                     'from': edge_attrs['from'],
                     'to': edge_attrs['to'],
                     'priority': edge_attrs['priority'],
                     'type': edge_attrs['type'],
                     'shape': lane['shape'],
                     'length': lane['length'],
                     'speed': lane['speed'],
                     'lane_id': lane['id'],
                     'allow_taxi': allow_taxi,
        }
        edges.append(edge_dict)
df = pd.DataFrame(edges)
df.to_csv('../data/traffic_flow/edges.csv')
# %%
from_set = set(df['from'].unique())
to_set = set(df['to'].unique())
print(len(from_set | to_set))
junctions = []
for junction in root.iter('junction'):
    j_attrs = junction.attrib
    if j_attrs['type'] != 'internal':
        junctions.append(j_attrs)
junction_df = pd.DataFrame(junctions)
junction_df.to_csv('../data/traffic_flow/nodes.csv')

# %%
from pyproj import Proj
netOffset = [-385174.69, -6671039.23] 
convBoundary = [0.00, 0.00, 1135.02, 1149.85] 
origBoundary = [24.914235, 60.156132, 24.966609, 60.189319]
proj = Proj("+proj=utm +zone=35 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
lats, lons = [], []

junction_df['x'] = junction_df['x'].astype(float)
junction_df['y'] = junction_df['y'].astype(float)

for e, row in df.iterrows():
    from_node = junction_df[junction_df['id'] == row['from']].iloc[0]
    to_node = junction_df[junction_df['id'] == row['to']].iloc[0]
    (lon1, lat1) = proj(from_node['x'] - netOffset[0], from_node['y'] - netOffset[1], inverse=True)
    (lon2, lat2) = proj(to_node['x'] - netOffset[0], to_node['y'] - netOffset[1], inverse=True)
    lats.append(lat1)
    lons.append(lon1)
df['lat'] = lats
df['lon'] = lons
df.to_csv('../data/traffic_flow/edges.csv')

# %%
# Split the region into N x N grids
N = 4
lon_min, lat_min, lon_max, lat_max = df['lon'].min(), df['lat'].min(), df['lon'].max() + 0.0001, df['lat'].max() + 0.0001
wlat = (lat_max - lat_min) / N
wlon = (lon_max - lon_min) / N
get_grid = lambda r: int((r['lat'] - lat_min) / wlat) * N + int((r['lon'] - lon_min) / wlon)
df['grid'] = df.apply(get_grid, axis=1)
df.describe()

# %%
cropped_master_data = master_data[(master_data['start_lat'] >= lat_min) &
                                  (master_data['start_lon'] >= lon_min) &
                                  (master_data['start_lat'] <= lat_max) &
                                  (master_data['start_lon'] <= lon_max)]
cropped_master_data['lat'] = cropped_master_data['start_lat']
cropped_master_data['lon'] = cropped_master_data['start_lon']
cropped_master_data['grid'] = cropped_master_data.apply(get_grid, axis=1)
cropped_master_data.describe()
selected_data = selected_data[list(cropped_master_data.index)]
selected_data.to_pickle('../data/traffic_flow/cropped_data.pkl')
# %%
start_t1, end_t1 = 360, 420
start_t2, end_t2 = 1320, 1380
selected_data.index = list(range(0, 1440))
cropped_master_data['p1'] = selected_data.loc[start_t1:end_t1].mean(axis=0)
cropped_master_data['p2'] = selected_data.loc[start_t2:end_t2].mean(axis=0)
# %%
cropped_master_data.describe()
# %%
import geopandas as gpd
from shapely import geometry
tessellation = []  # id, p1, p2, location
for i in range(N * N):
    temp = cropped_master_data[cropped_master_data['grid'] == i]
    yi = i // N
    xi = i % N
    geo = geometry.Polygon([(lon_min + xi * wlon, lat_min + yi * wlat), 
                            (lon_min + (xi + 1) * wlon, lat_min + yi * wlat), 
                            (lon_min + (xi + 1) * wlon, lat_min + (yi + 1) * wlat), 
                            (lon_min + xi * wlon, lat_min + (yi + 1) * wlat)])
    
    tessellation.append([i, temp['p1'].mean(), temp['p2'].mean(), geo])
tessellation = pd.DataFrame(tessellation, columns=['tile_id', 'p1', 'p2', 'geometry'])
tessellation = tessellation.fillna(0.001)
tessellation = gpd.GeoDataFrame(tessellation, geometry='geometry')
tessellation

# %%
edges = df[df['allow_taxi'] == 1]
import networkx as nx
G = nx.DiGraph()
nodes = set(edges['from'].unique()) | set(edges['to'].unique())
for node in nodes:
    G.add_node(node)
for i, e in edges.iterrows():
    G.add_edge(e['from'], e['to'])
len(nodes)

# 连通性检验
connected_pairs = {}
for u in nodes:
    connect_nodes = []
    shortest_paths = nx.shortest_path(G, u)
    for key, value in shortest_paths.items():
        if key != u:
            connect_nodes.append(key)
    connected_pairs[u] = connect_nodes

# 计算每个边的连通节点数
f = lambda x: len(connected_pairs[x])
edges['connected_cnt'] = edges['from'].apply(f)

for u in nodes:
    print(len(connected_pairs[u]))

# %%
# Generate Grid Trajectories
import skmob
from skmob.models.epr import DensityEPR
start_time = pd.to_datetime('2022/01/15 06:00:00')
end_time = pd.to_datetime('2022/01/15 07:00:00')
depr = DensityEPR()
num_usr = 900
# gmodel = skmob.models.gravity.Gravity(origin_exp=1.0, destination_exp=1.0)
tdf = depr.generate(start_time, end_time, tessellation, relevance_column='p1', n_agents=10000, show_progress=True)
print(tdf.head())
# tdf.plot_trajectory()

# Generate trips
f = lambda x: int((x['lat'] - lat_min) / wlat) * N + int((x['lng'] - lon_min) / wlon)
tdf['grid'] = tdf.apply(f, axis=1)

from xml.dom.minidom import Document
import random

doc = Document()

routes = doc.createElement('routes')
routes.setAttribute('xmlns:xsi', "http://www.w3.org/2001/XMLSchema-instance")
routes.setAttribute('xsi:noNamespaceSchemaLocation', 'http://sumo.dlr.de/xsd/routes_file.xsd')
doc.appendChild(routes)

trip_id = 0
usrs = tdf['uid'].unique()

trips = []
uid = 0
import numpy as np
for u in usrs:
    usr_traj = tdf[tdf['uid'] == u]
    from_edge = None
    depart = None

    for i, t in usr_traj.iterrows():
        grid = t['grid']
        grid_edges = edges[edges['grid'] == grid]
        if grid_edges.shape[0] == 0:
            continue
        
        if from_edge is not None:
            to_node = edges[edges['id'] == from_edge]['to'].values[0]
            connected_nodes = connected_pairs[to_node]
            connected_edges = grid_edges[grid_edges['from'].isin(connected_nodes)]
            if connected_edges.shape[0] == 0:
                continue
            else:
                e = connected_edges.sample(1)
                trips.append([trip_id, depart, from_edge, e.iloc[0]['id'], uid])
                from_edge = e.iloc[0]['id']

        else: 
            temp = grid_edges[grid_edges['connected_cnt'] > 1]
            if temp.shape[0] == 0:
                continue
            else:
                from_edge = temp.sample(1).iloc[0]['id']

        trip_id += 1
        dt = t['datetime']
        # depart = dt.hour * 3600 * 0 + dt.minute * 60 + dt.second
        depart = round(np.random.uniform(0, 1800), 1)
    
    if len(trips) > num_usr:
        break

    uid += 1
    



trips = pd.DataFrame(trips, columns=['id', 'depart', 'from', 'to', 'usr'])
trips = trips.sort_values(by='depart')
trips['id'] = list(range(trips.shape[0]))
for i, row in trips.iterrows():
    trip = doc.createElement('trip')
    trip.setAttribute('id', str(row['id']))
    trip.setAttribute('depart', str(row['depart']))
    trip.setAttribute('from', str(row['from']))
    trip.setAttribute('to', str(row['to']))
    routes.appendChild(trip)


with open('../data/traffic_flow/{}vehicles.trips.xml'.format(num_usr), 'w') as f:
    f.write(doc.toprettyxml(indent=' '))



# %%

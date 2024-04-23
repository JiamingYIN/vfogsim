# %%
import pickle as pkl
import pandas as pd
import numpy as np
import os
import random
# %%
dir = '../data/simple_helsinki'

MIN_LENGTH = 10

edges = pd.read_csv(os.path.join(dir, 'new_edges.csv'), dtype={'from': str, 'to': str, 'end_edge': str})
nodes = pd.read_csv(os.path.join(dir, 'new_nodes.csv'), dtype={'x': float, 'y': float, 'id': str})
# edges = pd.read_pickle(os.path.join(dir, 'new_edges.pkl'))
# edges.columns = ['from', 'to', 'length', 'node_list', 'start_edge', 'end_edge', 'detector', 'edge_list']
# nodes = pd.read_pickle(os.path.join(dir, 'new_nodes.pkl'))
# nodes.columns = ['x', 'y', 'id']


delta_x = []
delta_y = []
for i, row in edges.iterrows():
    temp = nodes[nodes['id'] == row['from']]
    x1, y1 = nodes[nodes['id'] == row['from']].iloc[0].values[:2]
    x2, y2 = nodes[nodes['id'] == row['to']].iloc[0].values[:2]
    delta_x.append(x2 - x1)
    delta_y.append(y2 - y1)
edges['delta_x'] = delta_x
edges['delta_y'] = delta_y
# %%
# id_map = dict(zip(edges['id'], range(edges.shape[0])))
def cos(e1, e2):
    return e1.dot(e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))

# %%
# Go straight, turn left, turn right, stay, U-turn
action_map = np.ones((edges.shape[0], 5)) * (-1)
edges['id'] = list(range(edges.shape[0]))

for i, row in edges.iterrows():
    neighbors = edges[edges['from'] == row['to']]
    e1 = np.array([row['delta_x'], row['delta_y']])
    action_map[i][3] = i

    if neighbors.shape[0] < 1:
        print("No need to select action!")
    for j, nrow in neighbors.iterrows():
        e2 = np.array([nrow['delta_x'], nrow['delta_y']])
        ct = cos(e1, e2)
        cp = np.cross(e1, e2)
        action_id = nrow['id']

        if nrow['length'] <= MIN_LENGTH:  # Very short edges

            nbrs_se = edges[edges['from'] == nrow['to']]
            long_nbrs = []

            for _, nbr in nbrs_se.iterrows():
                e3 = np.array([nbr['delta_x'], nbr['delta_y']])
                if nbr['length'] >= MIN_LENGTH:
                    long_nbrs.append(nbr['id'])
                    if cos(e2, e3) > 0.7071:  # Straight road of the short edge
                        action_id = nbr['id']

            if action_id == nrow['id']:  # No straight road
                print("No straight!")
                action_id = random.choice(long_nbrs) if len(long_nbrs) > 1 else -1

        if ct < -0.7071: 
            action_map[i][4] = action_id  # U-ture
        elif ct > 0.7071:
            action_map[i][0] = action_id  # Go straight
        else:
            if cp > 0:
                action_map[i][1] = action_id  # Turn left
            else:
                action_map[i][2] = action_id  # Turn right

test_map = (action_map == -1).sum(axis=1)
wrong_id = np.where(test_map == 4)[0]
print("Wrong edge:", wrong_id)
# %%
df = pd.DataFrame(action_map, columns=['straight', 'left', 'right', 'stay', 'Uturn'])
df = df.astype(int)
df = df.astype(str)
df.index = edges['end_edge']
# df['end_edge'] = edges['end_edge']
df.to_csv(os.path.join(dir, 'action_map.csv'))

# %%

import pandas as pd
import pickle as pkl
import numpy as np
import os

new_edges = pd.read_csv('../data/simple_helsinki/new_edges.csv', index_col=0)
edges = pd.read_csv('../data/sumo_20vehicles/plain.edg.csv', sep=';',
                    dtype={'edge_from': str, 'edge_to': str, 'edge_id': str})

node_lists = []
edge_lists = []
for i, row in new_edges.iterrows():
    nodelist = row['node_list']
    nodelist = [str(x.strip()[1:-1]) for x in nodelist[1:-1].split(',')]
    n1 = nodelist[0]
    new_nodelist = n1
    edgelist = ''
    for n2 in nodelist[1:]:
        edge_id = edges[(edges['edge_from'] == n1) & (edges['edge_to'] == n2)].iloc[0]['edge_id']
        print(edge_id)
        n1 = n2
        # edgelist.append(edge_id)
        edgelist += edge_id + ' '
        new_nodelist += ' ' + n2
    edge_lists.append(edgelist.strip())
    node_lists.append(new_nodelist)

new_edges['edge_list'] = edge_lists
new_edges['node_list'] = node_lists
new_edges = new_edges[['from', 'to', 'length', 'node_list', 'start_edge', 'end_edge', 'detector', 'edge_list']]

new_edges.to_csv('../data/simple_helsinki/new_edges.csv')

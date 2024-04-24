import pandas as pd
import numpy as np
import os

path = '../data/simple_helsinki'

edges = pd.read_csv(os.path.join(path, 'new_edges.csv'), index_col=0)
with open(os.path.join(path, 'routes.csv'), 'w') as f:
	for i, row in edges.iterrows():
		rid = 'route_' + row['end_edge']
		edges = row['edge_list']
		route = '<route id="{}" edges="{}"/>\n'.format(rid, edges)
		f.write(route)

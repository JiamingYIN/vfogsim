import pickle as pkl
import pandas as pd
import os


def load_data(path):
    data = {}
    edges = pd.read_csv(os.path.join(path, 'new_edges.csv'), index_col=0)
    print(edges.columns)

    data['edges'] = edges
    data['edge_ids'] = list(edges['end_edge'])
    data['detectors'] = list(edges['detector'])

    road_map = {}
    for i, row in edges.iterrows():
        edge_list = row['edge_list'].split(' ')
        for e in edge_list:
            road_map[e] = row['end_edge']
    data['road_map'] = road_map

    with open(os.path.join(path, 'new_nodes.pkl'), 'rb') as file:
        nodes = pkl.load(file)
    nodes.columns = ['x', 'y', 'id']
    # nodes.to_csv(os.path.join(path, 'new_nodes.csv'))
    data['nodes'] = nodes

    action_map = pd.read_csv(os.path.join(path, 'action_map.csv'), index_col=0)
    # index <==> road id
    id_map = dict(zip(range(edges.shape[0]), list(edges['end_edge'])))
    data['road_id_map'] = id_map
    data['action_map'] = action_map

    # TODO: Read route file
    data['user_ids'] = [str(x * 1.0) for x in range(2, 599)]
    # data['user_ids'] = ['10.0', '2.0', '3.0', '4.0', '5.0', '6.0']

    return data




if __name__ == '__main__':
    data = load_data('../data/simple_helsinki')

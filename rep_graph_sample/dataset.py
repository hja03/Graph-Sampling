from ogb.nodeproppred import NodePropPredDataset
from ogb.linkproppred import LinkPropPredDataset
import networkx as nx
import pickle
from torch_geometric.utils.convert import to_networkx


def ogb_dataset_to_nx_graph(dataset_name: str) -> nx.Graph:
    assert(dataset_name.startswith('ogb'))

    if dataset_name.startswith('ogbn'):
        dataset = NodePropPredDataset(name=dataset_name, root='./dataset')
        graph = dataset[0][0]

    elif dataset_name.startswith('ogbl'):
        dataset = LinkPropPredDataset(name=dataset_name, root='./dataset')
        graph = dataset[0]

    else:
        raise NotImplementedError(f'Dataset name: {dataset_name} not supported!')

    edges = ((e[0], e[1]) for e in graph['edge_index'].T)
    nx_graph = nx.from_edgelist(edges)

    if graph['node_feat'] is not None:
        node_features = {i:feature for i, feature in enumerate(graph['node_feat'])}
        nx.set_node_attributes(nx_graph, node_features, name='x')
    else:
        print('INFO: Dataset does not contain node features or they could not be found.')

    assert(nx_graph.number_of_nodes() == graph['num_nodes'])

    return nx_graph


def load_from_pickle(path: str, remove_data: bool = True):
    data = pickle.load(open(path, 'rb'))[0]

    if remove_data:
        if data.x is not None: del(data.x)
        if data.y is not None: del(data.y)

    g = to_networkx(data, to_undirected=True, remove_self_loops=True)
    return g
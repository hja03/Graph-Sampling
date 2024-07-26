import random
import numpy as np
from rep_graph_sample import SubgraphHandler, RunHistory, ogb_dataset_to_nx_graph


dataset_name = 'ogbn-arxiv'
subgraph_size = 100
num_iters = 1_000
exponent = 130


# Load Dataset
graph = ogb_dataset_to_nx_graph(dataset_name)


# Get an initial subgraph node set
subgraph_nodes = list(np.random.choice(graph.nodes(), size=(subgraph_size), replace=False))
#subgraph_nodes = list(np.load('0.151-60k.npy'))

subgraph = SubgraphHandler(
    full_graph = graph, 
    initial_node_set = subgraph_nodes
)

logger = RunHistory(subgraph_handler=subgraph,
                    save_interval=100,
                    p=exponent)

# Main Loop
prev_ks_dist = subgraph.ks_distance()

for i in range(num_iters):
    # Randomly pick a node to remove and add to the subgraph
    remove_node = np.random.choice(subgraph.nodes)
    add_node = np.random.choice(subgraph.nodes_not_in_subgraph)

    subgraph.remove(remove_node)
    subgraph.add(add_node)

    # Calculate the new Degree KS Distance
    new_ks_dist = subgraph.ks_distance()

    ratio = prev_ks_dist / new_ks_dist
    if ratio >= 1.0:
        # Accept
        prev_ks_dist = new_ks_dist
        accepted = True
    else:
        score = ratio ** exponent
        prop = random.random()
        if score > prop:
            # Accept
            prev_ks_dist = new_ks_dist
            accepted = True
        else:
            # Reject and revert back
            subgraph.add(remove_node)
            subgraph.remove(add_node)
            accepted = False

    logger.log(ratio=ratio, accepted=accepted)

logger.save_run()
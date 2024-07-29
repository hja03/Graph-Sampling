import random
import numpy as np
from rep_graph_sample import SubgraphHandler, RunHistory, ogb_dataset_to_nx_graph
from tqdm import tqdm
from argparse import ArgumentParser


# Parse Command Line Arguments
parser = ArgumentParser(
            prog='Metropolis-Hastings Graph Sampler'
         )

parser.add_argument('--subgraph-size', type=int, default=100)
parser.add_argument('--iters', type=int, default=1_000)
parser.add_argument('--exponent', type=int, default=100)
parser.add_argument('--dataset', type=str, default='ogbn-arxiv')

args = parser.parse_args()


# Load Dataset
print('Loading dataset.')
graph = ogb_dataset_to_nx_graph(args.dataset)


# Get an initial subgraph node set
subgraph_nodes = list(np.random.choice(graph.nodes(), size=(args.subgraph_size), replace=False))
#subgraph_nodes = list(np.load('0.151-60k.npy'))

subgraph = SubgraphHandler(
    full_graph = graph, 
    initial_node_set = subgraph_nodes
)

logger = RunHistory(subgraph_handler=subgraph,
                    save_interval=100,
                    p=args.exponent)

# Main Loop
prev_ks_dist = subgraph.ks_distance()

for i in tqdm(range(args.iters)):
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
        score = ratio ** args.exponent
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
print('Saved and done!')
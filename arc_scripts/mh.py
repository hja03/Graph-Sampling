import random
import numpy as np
from rep_graph_sample import SubgraphHandler, RunHistory, ogb_dataset_to_nx_graph, load_from_pickle
from tqdm import tqdm
from argparse import ArgumentParser


# Parse Command Line Arguments
parser = ArgumentParser(
            prog='Metropolis-Hastings Graph Sampler'
         )

parser.add_argument('--subgraph-size', type=int, default=100)
parser.add_argument('--iters', type=int, default=1_000)
parser.add_argument('--exponent', type=int, default=100)
parser.add_argument('--max-step-size', type=int, default=1)
parser.add_argument('--dataset', type=str, default='ogbn-arxiv')
parser.add_argument('--save-interval', type=int, default=100)

args = parser.parse_args()


# Load Dataset
print(f'Loading dataset: {args.dataset}')
if args.dataset == 'ogbn-arxiv':
    graph = ogb_dataset_to_nx_graph(args.dataset)
elif args.dataset == 'tiger':
    graph = load_from_pickle('./dataset/Large_Tiger_Alaska_90k.pkl')


# Get an initial subgraph node set
subgraph_nodes = list(np.random.choice(graph.nodes(), size=(args.subgraph_size), replace=False))
#subgraph_nodes = list(np.load('0.151-60k.npy'))

subgraph = SubgraphHandler(
    full_graph = graph, 
    initial_node_set = subgraph_nodes
)

logger = RunHistory(subgraph_handler=subgraph,
                    save_interval=args.save_interval,
                    p=args.exponent,
                    max_step_size=args.max_step_size)

# Main Loop
prev_ks_dist = subgraph.ks_distance()

for i in tqdm(range(args.iters)):
    # Randomly pick a node to remove and add to the subgraph
    num_nodes_to_change = np.random.randint(1, args.max_step_size + 1)
    remove_nodes = np.random.choice(subgraph.nodes, size=(num_nodes_to_change,), replace=False)
    add_nodes = np.random.choice(subgraph.nodes_not_in_subgraph, size=(num_nodes_to_change,), replace=False)

    for node in remove_nodes:
        subgraph.remove(node)
    for node in add_nodes:
        subgraph.add(node)

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
            for node in add_nodes:
                subgraph.remove(node)
            for node in remove_nodes:
                subgraph.add(node)
            accepted = False

    logger.log(ratio=ratio, accepted=accepted)

logger.save_run()
print('Saved and done!')
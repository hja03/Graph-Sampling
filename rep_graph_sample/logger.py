import numpy as np
import matplotlib.pyplot as plt
from .graph import SubgraphHandler
from pathlib import Path
from torch_geometric.utils.convert import from_networkx
import torch
from tqdm import tqdm


class RunHistory:
    def __init__(self, subgraph_handler: SubgraphHandler, save_interval: int, 
                 p: int, max_step_size: int, save_name: str = None) -> None:
        self.distances = []
        self.ratios = []
        self.accept_rejects = []

        self.saved_subgraphs = np.array(subgraph_handler.subgraph_nodes, dtype=np.int32)

        self.subgraph_handler = subgraph_handler

        self.iteration = 0
        self.save_interval = save_interval

        self.p = p
        self.max_step_size = max_step_size

        self.save_name = save_name

    def log(self, ratio: float, accepted: bool) -> None:
        self.iteration += 1

        self.distances.append(self.subgraph_handler.ks_distance())
        self.ratios.append(ratio)
        self.accept_rejects.append(accepted)

        if self.iteration % self.save_interval == 0:
            self.saved_subgraphs = np.vstack((self.saved_subgraphs, np.array(self.subgraph_handler.subgraph_nodes)))


    def test_exponent(self, exponent: int, num_repeats: int = 5, iter_range: tuple[int] = None) -> float:
        if iter_range is None:
            ratios = self.ratios
        else:
            ratios = self.ratios[iter_range[0]:iter_range[1]]

        accepted = 0
        acceptance_props = np.array(ratios) ** exponent
        for _ in range(num_repeats):
            random_nums = np.random.random(size=(len(ratios),))
            accepted += np.count_nonzero(random_nums < acceptance_props)

        acceptance_ratio = accepted / (len(ratios) * num_repeats)
        return acceptance_ratio
    

    def save_run(self) -> None:
        if self.save_name is not None:
            self.save_id = self.save_name
        else:
            if len(self.distances) >= 1000:
                self.save_id = f'p{self.p}_i{len(self.distances) // 1_000}k_n{self.max_step_size}_s{self.subgraph_handler.subgraph_size}'
            else:
                self.save_id = f'p{self.p}_i{len(self.distances)}_n{self.max_step_size}_s{self.subgraph_handler.subgraph_size}'


        Path(f"./runs/{self.save_id}").mkdir(parents=True, exist_ok=True)
        np.save(f'./runs/{self.save_id}/distances', self.distances)
        np.save(f'./runs/{self.save_id}/ratios', self.ratios)
        np.save(f'./runs/{self.save_id}/accept_rejects', self.accept_rejects)
        np.save(f'./runs/{self.save_id}/samples', self.saved_subgraphs)


    def load_run(self, folder_path: str) -> None:
        self.distances = np.load(f'{folder_path}/distances.npy')
        self.ratios = np.load(f'{folder_path}/ratios.npy')
        self.accept_rejects = np.load(f'{folder_path}/accept_rejects.npy')
        self.saved_subgraphs = np.load(f'{folder_path}/samples.npy')


    def export_samples(self) -> None:
        if self.save_name is not None:
            self.save_id = self.save_name
        else:
            if len(self.distances) >= 1000:
                self.save_id = f'p{self.p}_i{len(self.distances) // 1_000}k_n{self.max_step_size}_s{self.subgraph_handler.subgraph_size}'
            else:
                self.save_id = f'p{self.p}_i{len(self.distances)}_n{self.max_step_size}_s{self.subgraph_handler.subgraph_size}'

        Path(f"./runs/{self.save_id}/samples").mkdir(parents=True, exist_ok=True)
        for i in range(len(self.saved_subgraphs)):
            subgraph = self.subgraph_handler.full_graph.subgraph(self.saved_subgraphs[i])
            subgraph = from_networkx(subgraph)
            torch.save(subgraph, f'./runs/{self.save_id}/samples/sample_{i}')


    def plot_distances(self, avg_window_size:int = 1) -> None:
        plt.plot(np.convolve(self.distances, np.ones(avg_window_size,) / avg_window_size, mode='valid'))
        plt.xlabel('Iteration')
        plt.ylabel('KS Distance')


    def plot_degree_distributions(self, percentage_range: tuple[int] = None) -> None:
        plt.ecdf(self.subgraph_handler.full_graph_degree_sequence, label='True')

        percentage_range = percentage_range if percentage_range is not None else (0, 100)
        idx_range = (int(percentage_range[0] * len(self.saved_subgraphs) * 0.01), int(percentage_range[1] * len(self.saved_subgraphs) * 0.01) - 1)
        samples = self.saved_subgraphs[idx_range[0]:idx_range[1]]

        degrees = [list(dict(self.subgraph_handler.full_graph.subgraph(nodes).degree()).values()) for nodes in samples]
        all_degrees = []
        for d in degrees:
            all_degrees += d

        plt.ecdf(all_degrees, label='Mean Sample')

        plt.xscale('log')
        plt.legend()

        plt.xlabel('Degree')
        plt.ylabel('P(d < D)')


    def plot_sample_similarity(self, percentage_range: tuple[int] = None, subsample_factor: int = 1) -> None:
        percentage_range = percentage_range if percentage_range is not None else (0, 100)
        idx_range = (int(percentage_range[0] * len(self.saved_subgraphs) * 0.01), int(percentage_range[1] * len(self.saved_subgraphs) * 0.01) - 1)
        samples = self.saved_subgraphs[idx_range[0]:idx_range[1]:subsample_factor]

        node_sets = [set(nodes) for nodes in samples]
        edge_sets = [set(self.subgraph_handler.full_graph.subgraph(nodes).edges()) for nodes in samples]

        def jaccard_sim_matrix(set_list: list[set]):
            matrix = np.zeros((len(set_list), len(set_list)))

            for i, set_i in tqdm(enumerate(set_list), total=len(set_list)):
                for j, set_j in enumerate(set_list):
                    if len(set_i) == 0 or len(set_j) == 0:
                        matrix[i, j] = 0
                    elif i != j:
                        matrix[i,j] = len(set_i.intersection(set_j)) / len(set_i.union(set_j))
                    else:
                        matrix[i,j] = 1
            return matrix


        fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        node_image = axs[0].imshow(jaccard_sim_matrix(node_sets), vmin=0, vmax=1, origin='lower')
        fig.colorbar(node_image, ax=axs[0])

        axs[0].set_xlabel('Sampled Subgraph #')
        axs[0].set_ylabel('Sampled Subgraph #')
        axs[0].set_title('Jaccard - Nodes')


        edge_image = axs[1].imshow(jaccard_sim_matrix(edge_sets), vmin=0, vmax=1, origin='lower')
        fig.colorbar(edge_image, ax=axs[1])

        axs[1].set_xlabel('Sampled Subgraph #')
        axs[1].set_title('Jaccard - Edges')

        fig.suptitle('Diversity of Samples')


    def plot_acceptance_ratio(self, avg_window_size: int = 200) -> None:
        fig, ax = plt.subplots(1,1)
        
        ax.plot(np.convolve(self.accept_rejects, np.ones(avg_window_size,) / avg_window_size, mode='valid'))

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Proportion of Accepted Samples')
        ax.set_ylim(bottom=0, top=1)
        ax.grid(axis='y', linestyle='dashed')

    @property
    def iterations(self):
        return len(self.distances)

    @property
    def acceptance_ratio(self):
        return np.count_nonzero(self.accept_rejects) / len(self.distances)

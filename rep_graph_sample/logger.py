import numpy as np
import matplotlib.pyplot as plt
from .graph import SubgraphHandler
from datetime import datetime
from pathlib import Path


class RunHistory:
    def __init__(self, subgraph_handler: SubgraphHandler, save_interval: int, p: int) -> None:
        self.distances = []
        self.ratios = []
        self.accept_rejects = []

        self.saved_subgraphs = np.array(subgraph_handler.subgraph_nodes, dtype=np.int32)

        self.subgraph_handler = subgraph_handler

        self.iteration = 0
        self.save_interval = save_interval

        self.p = p


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
        self.save_id = f'p{self.p}_i{len(self.distances)}'

        Path(f"./runs/{self.save_id}").mkdir(parents=True, exist_ok=True)
        np.save(f'./runs/{self.save_id}/distances', self.distances)
        np.save(f'./runs/{self.save_id}/ratios', self.ratios)
        np.save(f'./runs/{self.save_id}/accept_rejects', self.accept_rejects)
        np.save(f'./runs/{self.save_id}/samples', self.saved_subgraphs)


    def plot_distances(self) -> None:
        plt.plot(self.distances)
        plt.xlabel('Iteration')
        plt.ylabel('KS Distance')

    @property
    def iterations(self):
        return len(self.distances)

    @property
    def acceptance_ratio(self):
        return np.count_nonzero(self.accept_rejects) / len(self.distances)

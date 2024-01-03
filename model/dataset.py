# Paper Title: A Second Look on BASS - Boosting Abstractive Summarization with Unified Semantic Graphs
# Author: Osman Alperen Koras
# Affiliation: Institute for AI in Medicine (IKIM), University Medicine Essen
# Email: osman.koras@uni-due.de
# Date: 12.10.2023
#
# License: MIT
import gzip
import pathlib
import pickle
import time

from torch.utils.data import Dataset


class BassDataset(Dataset):
    def __init__(self, dataset_dir: pathlib.Path, split, device="cpu", load_in_memory=False):
        self.load_in_memory = load_in_memory
        
        start_time = time.time()
        self.device = device
        self.dataset_dir = dataset_dir
        files = [f.relative_to(dataset_dir) for f in (dataset_dir / split).rglob("*.gz/[0-9]*")]

        if self.load_in_memory:
            self.datapoints = []
            for file in files:
                with gzip.open(self.dataset_dir / file, "rb") as fin:
                    dp = pickle.load(fin)
                    self.datapoints.append(dp)
        else:
            self.datapoints = files
            
        end_time = time.time()
        print(f"loading the dataset took {end_time - start_time} seconds...")

    def __len__(self):
        if self.load_in_memory:
            return len(self.datapoints)
        else:
            return len(self.datapoints)

    def __getitem__(self, idx):
        if self.load_in_memory:
            dp = self.datapoints[idx]
            return dp.input_tokens, dp.input_padding, dp.output_tokens, dp.output_padding, dp.graph_construction_matrix, dp.node_padding, dp.graph_adjacency_matrix
        
        else:
            with gzip.open(self.dataset_dir / self.datapoints[idx], "rb") as fin:
                dp = pickle.load(fin)
            
            return dp.input_tokens, dp.input_padding, dp.output_tokens, dp.output_padding, dp.graph_construction_matrix, dp.node_padding, dp.graph_adjacency_matrix

        

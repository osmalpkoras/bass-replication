# Paper Title: A Second Look on BASS - Boosting Abstractive Summarization with Unified Semantic Graphs
# Author: Osman Alperen Koras
# Affiliation: Institute for AI in Medicine (IKIM), University Medicine Essen
# Email: osman.koras@uni-due.de
# Date: 12.10.2023
#
# License: MIT
from __future__ import annotations

import gzip
import os
import pickle

import torch

from preprocessing import doc


class Dataset:
    def __init__(self):
        self.files: list[DatasetFile] = []

    def save(self, file_path):
        with gzip.open(file_path, "wb+") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path) -> Dataset:
        dataset = None
        try:
            if os.path.isfile(file_path):
                with gzip.open(file_path, "rb+") as f:
                    dataset = pickle.load(f)
        except Exception:
            dataset = None

        return dataset


    def add(self, path, datapoint):
        datafile = None
        for file in self.files:
            if file.path == path:
                datafile = file

        if datafile is None:
            datafile = DatasetFile()
            datafile.path = path
            self.files.append(datafile)

        if datapoint not in datafile.datapoints:
            datafile.datapoints.append(datapoint)

    def contains(self, path, datapoint):
        for file in self.files:
            if file.path == path:
                if datapoint in file.datapoints:
                    return True
                else:
                    return False
        return False

    def exclude(self, other_dataset: Dataset):
        if other_dataset is None:
            return

        for file in other_dataset.files:
            matching_files = [f for f in self.files if f.path == file.path]
            for f in matching_files:
                f.datapoints = [d for d in f.datapoints if d not in file.datapoints]

        self.files = [f for f in self.files if len(f.datapoints) > 0]

    def extend(self, other_dataset: Dataset):
        for file in other_dataset.files:
            matching_file = None
            for f in self.files:
                if f.path == file.path:
                    matching_file = f
                    break

            if matching_file is None:
                matching_file = DatasetFile()
                matching_file.path = file.path
                self.files.append(matching_file)

            matching_file.datapoints.extend([d for d in file.datapoints if d not in matching_file.datapoints])

        self.files = [f for f in self.files if len(f.datapoints) > 0]


class DatasetFile:
    def __init__(self):
        self.path: str = None
        self.datapoints: list[str] = []


class Datapoint:
    def __init__(self):
        self.id: str = None

    def filepath(self) -> str:
        return self.id.split(":")[0]

    def row_index(self) -> str:
        return self.id.split(":")[1]


class BassDatapoint(Datapoint):
    def __init__(self):
        super(BassDatapoint, self).__init__()
        self.input_tokens: torch.Tensor = None
        self.input_padding: torch.Tensor = None
        self.output_tokens: torch.Tensor = None
        self.output_padding: torch.Tensor = None
        self.graph_construction_matrix: torch.Tensor = None
        self.node_padding: torch.Tensor = None
        self.graph_adjacency_matrix: torch.Tensor = None


class CoreNlpDatapoint(Datapoint):

    @staticmethod
    def parse_from(obj: CoreNlpDatapoint, include_protobuf_annotation = False, include_documents=True):
        datapoint = CoreNlpDatapoint(str(obj.document_text), str(obj.summary_text), [CoreNlpDatapoint.parse_bytes(a) for a in obj.protobuf_annotation] if include_protobuf_annotation else [])
        datapoint.id = str(obj.id)
        datapoint.annotation = [doc.Document.parse_from(d) for d in obj.annotation] if include_documents else []
        return datapoint

    def __init__(self, input: str, output: str, annotation):
        super(CoreNlpDatapoint, self).__init__()
        self.document_text = input
        self.summary_text = output
        self.protobuf_annotation: list[bytes] = annotation
        self.annotation: list[doc.Document] = []

    def __str__(self):
        return f"input: {self.document_text}, output: {self.summary_text}"

    @staticmethod
    def parse_bytes(b: any):
        return b''.join([i.to_bytes(1, 'big', signed=True) for i in b.data])
